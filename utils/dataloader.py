import os
import torch
from torch.utils.data.dataset import Dataset
import cv2 as cv
from utils.utils import letter_box, image_pretreat, image_translation


class SimpleDatasetClass(Dataset):
    def __init__(self):
        super().__init__()
        self.img_prepath1 = 'E:/dataset/famousface'
        self.img_prepath0 = 'E:/dataset/face/face_train/0'
        self.img_name1 = os.listdir(self.img_prepath1)
        self.img_name0 = os.listdir(self.img_prepath0)

    def __len__(self):
        return len(self.img_name1) + len(self.img_name0)

    def __getitem__(self, item):
        if item < len(self.img_name1):
            img_path = self.img_prepath1 + '/' + self.img_name1[item]
        else:
            img_path = self.img_prepath0 + '/' + self.img_name0[item - len(self.img_name1)]
        image = cv.imread(img_path)
        image = image_pretreat(image, 224)
        label = 1 if item < len(self.img_name1) else 0
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class SimpleDatasetDetect(Dataset):
    def __init__(self):
        super().__init__()
        self.img_path = 'E:/dataset/face1.png'
        self.x = 138
        self.y = 88
        self.w = 86
        self.h = 136
        self.sx = 62
        self.sy = 38

    def __len__(self):
        return self.x * self.y

    def __getitem__(self, item):
        image = cv.imread(self.img_path)
        # image = image_pretreat(image, 224)
        image = letter_box(image, 224)
        move_y = int(item/self.x)
        move_x = item - move_y * self.x
        image = image_translation(image, move_x-self.sx, move_y-self.sy)
        image = image[:, :, ::-1].transpose(2, 0, 1)/256
        label = [self.sx-move_x, self.sy-move_y, self.w, self.h]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)/224
