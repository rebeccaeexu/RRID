from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
# from basicsr.metrics.niqe import calculate_niqe
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from vd.metrics.vd_metric import calculate_vd_psnr, calculate_vd_ssim

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_vd_psnr', 'calculate_vd_ssim']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
