import numpy as np
import cv2 as cv


def letter_box(img, new_shape):
    """将图片等比例缩放调整到指定边长的正方形,剩下的填充"""
    shape = img.shape[:2]  # [h, w]
    r = min(new_shape / shape[0], new_shape / shape[1])  # scale ratio (new / old)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    dw, dh = (new_shape - new_unpad[0]) / 2, (new_shape - new_unpad[1]) / 2  # wh padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=(255, 255, 255))  # add border
    return img


def letter_box_ractangle(img, new_shape=416):
    """将图片等比例缩放调整到指定大小且为32的倍数,剩下的填充
    :param img: 输入图像[h,w,c]
    :param new_shape: 图像长边的长度
    """
    shape = img.shape[:2]  # [h, w]
    r = min(new_shape / shape[0], new_shape / shape[1])  # scale ratio (new / old)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    dw, dh = (new_shape - new_unpad[0]) / 2, (new_shape - new_unpad[1]) / 2  # wh padding
    dw, dh = np.mod(dw, 16), np.mod(dh, 16)  # wh padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=(255, 255, 255))  # add border
    return img


def image_pretreat(img, shape):
    img = letter_box(img, shape)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    return img / 256


def image_translation(img, dx, dy):
    affine_arr = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv.warpAffine(img, affine_arr, (img.shape[0], img.shape[1]))
