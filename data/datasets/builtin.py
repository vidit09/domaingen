# Copyright (c) Facebook, Inc. and its affiliates.

from .diverse_weather import register_dataset as register_diverse_weather
from .pascal_voc_adaptation import register_all_pascal_voc as register_pascal_voc
from .comic_water_adaptation import register_dataset as register_comic_water
import os 

_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
DEFAULT_DATASETS_ROOT = "data/"


register_diverse_weather(_root)
register_pascal_voc(_root)
register_comic_water(_root)
