import torch
import numpy as np

def rand_bbox(size, lam):
    W = size[2]
    if len(size)>3:
        H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    if len(size)>3:
        cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    if len(size)>3:
        cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    if len(size)>3:
        bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    if len(size)>3:
        bby2 = np.clip(cy + cut_h // 2, 0, H)

    if len(size) > 3:
        return bbx1, bby1, bbx2, bby2
    else:
        return bbx1, bbx2


def cutmix_data(x, y, alpha = 1.0, use_cuda = True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    if len(x.size()) > 3:
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    else:
        bbx1, bbx2 = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    if len(x.size()) > 3:
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    else:
        mixed_x[:,:, bbx1:bbx2] = x[index, :, bbx1:bbx2]
    y_a, y_b = y, y[index]
    if len(x.size()) > 3:
        lam = ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    else:
        lam = (bbx2 - bbx1) / (x.size()[-1])
    return mixed_x, y_a, y_b, lam


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
