import os
from glob import glob


def load_data(path):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))
    return images, masks


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
