import importlib
from os import path as osp

from basicsr.utils import scandir

# automatically scan and import metric modules for registry
# scan all the files that end with 'metric.py' under the metrics folder
metric_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(metric_folder) if v.endswith('_metric.py')]
# import all the metric modules
_metric_modules = [importlib.import_module(f'vd.metrics.{file_name}') for file_name in metric_filenames]
