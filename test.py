# flake8: noqa
import os.path as osp

import vd.archs
import vd.data
import vd.models
import vd.metrics
import vd.losses
from basicsr.test import test_pipeline

if __name__ == '__main__':
    # root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    root_path = './'
    test_pipeline(root_path)
