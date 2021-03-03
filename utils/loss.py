import torch
import math


def bounding_box_loss(data, label):
    """
    边界框回归损失
    :param data: x,y,w,h
    :param label: x,y,w,h
    """
    x = torch.sigmoid(data[:, 0])
    y = torch.sigmoid(data[:, 1])
    w = 0.4 * torch.exp(data[:, 2])
    h = 0.6 * torch.exp(data[:, 3])
    return box_ciou(torch.stack([x, y, w, h], 1), label)


def box_ciou(b1, b2):
    """
    :param b1: shape=(batch, 4)
    :param b2: shape=(batch, 4),
    :return ciou: shape=(batch, 1)
    """
    # 求出预测框和真实框的左上角右下角
    b1_xy = b1[:, :2]
    b1_wh = b1[:, 2:4]
    b1_wh_half = b1_wh/2
    b1_x1y1 = b1_xy - b1_wh_half
    b1_x2y2 = b1_xy + b1_wh_half
    b2_xy = b2[:, :2]
    b2_wh = b2[:, 2:4]
    b2_wh_half = b2_wh/2
    b2_x1y1 = b2_xy - b2_wh_half
    b2_x2y2 = b2_xy + b2_wh_half
    # 求出预测框和真实框的IOU
    intersect_mins = torch.max(b1_x1y1, b2_x1y1)
    intersect_maxes = torch.min(b1_x2y2, b2_x2y2)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[:, 0] * intersect_wh[:, 1]
    b1_area = b1_wh[:, 0] * b1_wh[:, 1]
    b2_area = b2_wh[:, 0] * b2_wh[:, 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / union_area
    # 计算中心的差距
    center_distance = torch.sum((b1_xy - b2_xy)**2, dim=1)
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_x1y1, b2_x1y1)
    enclose_maxes = torch.max(b1_x2y2, b2_x2y2)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = torch.sum(enclose_wh**2, dim=1)
    ciou = iou - 1.0 * center_distance / enclose_diagonal
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[:, 0]/b1_wh[:, 1]) - torch.atan(b2_wh[:, 0]/b2_wh[:, 1])), 2)
    alpha = v / (1.0 - iou + v)
    ciou = ciou - alpha * v
    return ciou


def box_iou(box1, box2):
    pass
