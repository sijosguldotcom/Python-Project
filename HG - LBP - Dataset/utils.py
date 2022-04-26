import os
import cv2
import numpy as np


def collect_dataset():
    images = []
    labels = []
    label_dic = {}

    BASE_PATH = os.path.dirname(__file__) + '/'
    datasets = [dataset for dataset in os.listdir(BASE_PATH + 'dataset/')]

    for i, dataset in enumerate(datasets):
        label_dic[i] = dataset
        for image in sorted(os.listdir(BASE_PATH + 'dataset/' + dataset)):
            images.append(cv2.imread(BASE_PATH + 'dataset/' + dataset + '/' +
                                     image, 0))
            labels.append(i)

    return (images, np.array(labels), label_dic)
