import os
from .dataset import DataLoaderTrain, DataLoaderVal

def get_training_data(rgb_dir, ratio, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, ratio, img_options, None)

def get_validation_data(rgb_dir, ratio):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, ratio, None)


