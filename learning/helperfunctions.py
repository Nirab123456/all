import pathlib
from typing import Tuple,Dict,List
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset


def find_classes(directory:str)->Tuple[List[str],Dict[str,int]]:
    """
    Returns all the class names in a given directory.
    """
    classes = [item.name for item in pathlib.Path(directory).glob('*') if item.is_dir()]

    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}

    return classes, class_to_idx


def show_random_image(dataset:Dataset):
    """show random image from dataset"""
    index = random.randint(0, len(dataset)-1)
    img, label = dataset[index]
    plt.imshow(img.permute(1,2,0))
    plt.title(f'label: {dataset.classes[label]} ||||index: {label}\n {img.shape}')
    plt.axis(False)
    plt.show()


def show_batch_images(dataloader: Dataset, classes: list[str] = None, n: int = 10, display_shape: bool = True):
    """Show batch images"""
    for i in range(n):
        index = random.randint(0, len(dataloader)-1)
        img, label = dataloader[index]
        if display_shape:
            print(f'img shape: {img.shape}')
        if classes:
            print(f'label: {classes[label]}')
        else:
            print(f'label: {label}')
        plt.imshow(img.permute(1,2,0))
        plt.axis('off')
        plt.show()