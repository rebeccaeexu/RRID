import torch
import os.path as osp
import time
from tqdm import tqdm
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs import build_network
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger
from basicsr.losses import build_loss
from basicsr.metrics.psnr_ssim import calculate_ssim_pt, calculate_psnr_pt
from vd.metrics.vd_metric import calculate_vd_ssim, calculate_vd_psnr

from vd.data.data_util import tensor2numpy, imwrite_gt

from deepspeed.profiling.flops_profiler import get_model_profile


@MODEL_REGISTRY.register()
class RAWRGBIDModel(SRModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define networks
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # self.net_f = build_network(self.opt['network_f'])
        # self.net_f = self.model_to_device(self.net_f)
        # self.print_network(self.net_f)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # # load pretrained models
        # load_path_f = self.opt['path'].get('pretrain_network_f', None)
        # if load_path_f is not None:
        #     param_key_f = self.opt['path'].get('param_key_f', 'params')
        #     self.load_network(self.net_f, load_path_f, self.opt['path'].get('strict_load_f', True), param_key_f)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        # self.net_f.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # self.net_f_ema = build_network(self.opt['network_f']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # for k, v in self.net_f.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
        #     else:
        #         logger = get_root_logger()
        #         logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output_rgb, self.output_raw = self.net_g(self.lq_rgb, self.lq_raw)
        # self.output1, align_feat = self.net_g(self.lq)
        # self.output = self.net_f(self.output1, align_feat)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix_rgb = self.cri_pix(self.output_rgb, self.gt_rgb)
            l_total += l_pix_rgb
            loss_dict['l_pix_rgb'] = l_pix_rgb

            l_pix_raw = self.cri_pix(self.output_raw, self.gt_raw)
            l_total += 0.5 * l_pix_raw
            loss_dict['l_pix_raw'] = l_pix_raw

            # l_inter = self.cri_pix(self.output1, self.gt)
            # l_total += l_inter
            # loss_dict['l_inter'] = l_inter
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output_rgb, self.gt_rgb)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        scale = self.opt.get('scale', 1)
        _, _, h_old, w_old = self.lq_rgb.size()

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            # self.net_f.eval()
            with torch.no_grad():
                self.output_rgb, self.output_raw = self.net_g_ema(self.lq_rgb, self.lq_raw)
                self.output_rgb = self.output_rgb[:, :, :h_old * scale, :w_old * scale]
                # self.output1, align_feat = self.net_g_ema(self.lq)
                # self.output1 = self.output1[:, :, :h_old * scale, :w_old * scale]
                # self.output = self.net_f(self.output1, align_feat)
                # self.output = self.output[:, :, :h_old * scale, :w_old * scale]
        else:
            self.net_g.eval()
            # self.net_f.eval()
            with torch.no_grad():
                self.output_rgb, self.output_raw = self.net_g(self.lq_rgb, self.lq_raw)
                self.output_rgb = self.output_rgb[:, :, :h_old * scale, :w_old * scale]
                # self.output1, align_feat = self.net_g(self.lq)
                # self.output1 = self.output1[:, :, :h_old * scale, :w_old * scale]
                # self.output = self.net_f(self.output1, align_feat)
                # self.output = self.output[:, :, :h_old * scale, :w_old * scale]
            self.net_g.train()
            # self.net_f.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
                # self.metric_results['SSIM_inter'] = 0
                # self.metric_results['PSNR_inter'] = 0
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        time_inf_total = 0.
        for idx, val_data in enumerate(dataloader):
            img_name = val_data['key'][0]
            self.feed_data(val_data)
            st = time.time()
            self.test()
            st1 = time.time() - st
            # print('The test time for this image %.3f' % st1)
            time_inf_total += st1

            visuals = self.get_current_visuals()
            # sr_img = tensor2numpy(visuals['result'])
            # metric_data['img'] = sr_img
            # if 'gt' in visuals:
            #     gt_img = tensor2numpy(visuals['gt'])
            #     metric_data['img2'] = gt_img
            #     del self.gt_rgb

            if self.opt['is_train']:
                sr_img_tensors = self.output_rgb.detach()
                metric_data['img'] = sr_img_tensors
                if 'gt' in visuals:
                    gt_img_tensors = self.gt_rgb.detach()
                    metric_data['img2'] = gt_img_tensors
                    del self.gt_rgb
            else:
                sr_img = tensor2numpy(visuals['result'])
                metric_data['img'] = sr_img
                if 'gt' in visuals:
                    gt_img = tensor2numpy(visuals['gt'])
                    metric_data['img2'] = gt_img
                    del self.gt_rgb

            # tentative for out of GPU memory
            del self.lq_rgb
            del self.lq_raw
            del self.output_rgb
            del self.output_raw
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    pass
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f'{img_name}.png')
                    imwrite_gt(sr_img, save_img_path)


            if with_metrics:
                # if self.opt['is_train']:
                #     self.metric_results['SSIM_inter'] += calculate_ssim_pt(self.output1.detach(),
                #                                                            metric_data['img2'],
                #                                                            0).detach().cpu().numpy().sum()
                #     self.metric_results['PSNR_inter'] += calculate_psnr_pt(self.output1.detach(),
                #                                                            metric_data['img2'],
                #                                                            0).detach().cpu().numpy().sum()
                # else:
                #     self.metric_results['SSIM_inter'] += calculate_vd_ssim(sr_img_inter,
                #                                                            metric_data['img2']).sum()
                #     self.metric_results['PSNR_inter'] += calculate_vd_psnr(sr_img_inter,
                #                                                            metric_data['img2']).sum()
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if self.opt['is_train']:
                        self.metric_results[name] += calculate_metric(metric_data, opt_).detach().cpu().numpy().sum()
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        time_avg = time_inf_total / (idx + 1)
        print('The average test time is %.3f' % time_avg)
        if with_metrics:
            for metric in self.metric_results.keys():
                if self.opt['is_train']:
                    self.metric_results[metric] /= 408
                else:
                    self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        # self.save_network(self.net_f, 'net_f', current_iter)
        self.save_training_state(epoch, current_iter)

    def compleity(self, batch_size):
        flops_g, macs_g, params_g = get_model_profile(model=self.net_g,  # model
                                                      input_shape=None,
                                                      # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                                      # args=None,  # list of positional arguments to the model.
                                                      args=[torch.zeros((1, 3, 512, 512), device=self.device),
                                                            torch.zeros((1, 4, 256, 256), device=self.device)],
                                                      kwargs=None,  # dictionary of keyword arguments to the model.
                                                      print_profile=True,
                                                      # prints the model graph with the measured profile attached to each module
                                                      detailed=True,  # print the detailed profile
                                                      module_depth=-1,
                                                      # depth into the nested modules, with -1 being the inner most modules
                                                      top_modules=1,
                                                      # the number of top modules to print aggregated profile
                                                      warm_up=10,
                                                      # the number of warm-ups before measuring the time of each module
                                                      as_string=True,
                                                      # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                                      output_file=None,
                                                      # path to the output file. If None, the profiler prints to stdout.
                                                      ignore_modules=None)  # the list of modules to ignore in the profiling
        return flops_g, macs_g, params_g
