import os
import torch
import cv2 as cv
from utils.utils import image_pretreat
from model.simplenet import SimpleNet
import shutil

model = SimpleNet()
model.load_state_dict(torch.load('resnet18.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Network loading complete.')
model.eval()
model.to(device)
with torch.no_grad():
    img_prepath = 'E:/dataset/face/face_test/1'
    img_name = os.listdir(img_prepath)
    for n, name in enumerate(img_name):
        img_path = img_prepath + '/' + name
        image = cv.imread(img_path)
        image = image_pretreat(image, 224)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        predict = model(image.to(device))
        if predict[0][0] > 0.4:
            print(name, predict[0][0])
            # shutil.move(img_path, 'E:/dataset/face/face_train/new')
# print(f'predict:{predict}')
