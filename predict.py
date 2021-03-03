import torch
import cv2 as cv
from utils.utils import image_pretreat
from model.simplenet import SimpleNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='su35.png')
opt = parser.parse_args()
model = SimpleNet()
model.load_state_dict(torch.load('resnet18.pth'))
image = cv.imread(opt.path)
image = image_pretreat(image, 224)
print('Network loading complete.')
model.eval()
with torch.no_grad():
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    predict = model(image)
    predict = torch.sigmoid(predict[0][0]).item()
print(f'predict:{predict}')
