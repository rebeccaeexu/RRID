import os
import os.path as osp
import glob
import cv2
import numpy as np

from basicsr.utils import scandir



def rawrgb_paired_paths_from_folders(folders):
    assert len(folders) == 4, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    input_rgb_folder, input_raw_folder, gt_rgb_folder, gt_raw_folder = folders

    gt_paths = list(scandir(gt_rgb_folder))
    gt_names = []
    for gt_path in gt_paths:
        gt_name = osp.basename(gt_path).split('_')[0]
        gt_names.append(gt_name)

    paths = []
    for gt_name in gt_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_name + '_gt.png')
        gt_raw_path = osp.join(gt_raw_folder, gt_name + '_gt.npz')
        lq_rgb_path = osp.join(input_rgb_folder, gt_name + '_m.png')
        lq_raw_path = osp.join(input_raw_folder, gt_name + '_m.npz')

        paths.append(dict(
            [('gt_rgb_path', gt_rgb_path), ('gt_raw_path', gt_raw_path), ('lq_rgb_path', lq_rgb_path),
             ('lq_raw_path', lq_raw_path), ('key', gt_name)]))
    return paths




def tensor2numpy(tensor):
    img_np = tensor.squeeze().numpy()
    img_np[img_np < 0] = 0
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    return img_np.astype(np.float32)


def imwrite_gt(img, img_path, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(img_path))
        os.makedirs(dir_name, exist_ok=True)

    img = img.clip(0, 1.0)
    uint8_image = np.round(img * 255.0).astype(np.uint8)
    cv2.imwrite(img_path, uint8_image)
    return None


def read_img(img_path):
    img = cv2.imread(img_path, -1)
    return img / 255.


def pack_rggb_raw(path):
    # pack RGGB Bayer raw to 4 channels
    raw_img = np.load(path)
    raw_data = raw_img['patch_data']
    im = raw_data / 4095.0

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :]), axis=2)
    return out
