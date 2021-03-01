import torch


def bounding_box_loss(data, label):
    """
    边界框回归损失
    :param data: p1,p2,p3,p4
    :param label: x,y,w,h
    """
    px = torch.sigmoid(data[0])
    py = torch.sigmoid(data[1])
    pw = 0.4 * torch.exp(data[2])
    ph = 0.6 * torch.exp(data[3])
    return box_ciou([px,py,pw,ph], label)


def box_ciou(box1, box2):
    dist = (box1 - box2) ** 2
    return sum(dist)
