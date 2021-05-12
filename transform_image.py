import PIL.Image
import cv2
import numpy as np

from lib import *

from argumentation import Compose, ConvertFromInts, ToAbsoluteCoords, \
    PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, \
    ToPercentCoords, Resize, SubtractMeans


class DataTransform():

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(color_mean),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ]),
            "val": Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


if __name__ == "__main__":
    color_mean = (104, 117, 123)
    input_size = 300
    transform = DataTransform(input_size, color_mean)
    a = [[0.1, 0.2, 0.3, 0.4, 5], [0.1, 0.2, 0.3, 0.4, 6]]
    a = np.array(a)
    phase = "train"
    for i in range(1,99):
        image_path = cv2.imread('./data/vinfast/vinfast ({}).jpg'.format(i))
        img_transformed, boxes, labels = transform(image_path, phase, a[:, :4], a[:, 4])
        image_transform=cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB)
        cv2.imwrite('data/data augmentation/vinfast_transform_{}.jpg'.format(i),image_transform)



