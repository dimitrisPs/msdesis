import torch 
import torch.nn as nn
from monai.losses.dice import DiceLoss


class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon:int=1e-5):
        super().__init__()
        self.epsilon=epsilon
    def forward(self, prediction, target):

        prediction = prediction.view(-1)
        target = target.view(-1)
        intersection = (prediction * target).sum()                            
        dice = (2.*intersection + self.epsilon)/(prediction.sum() + target.sum() + self.epsilon)
        return 1-dice

class SoftDiceLossWithLogits(SoftDiceLoss):
    def forward(self, prediction, target):
        return super().forward(torch.sigmoid(prediction), target)

class SoftJaccardLoss(nn.Module):
    def __init__(self, eps:int=1e-5):
        super().__init__()
        self.eps=eps
    def forward(self, prediction, target):

        prediction = prediction.view(-1)
        target = target.view(-1)

        intersection = (prediction * target).sum()
        union = prediction.sum() + target.sum()

        jaccard = (intersection + self.eps) / (union - intersection + self.eps)

        return 1- jaccard

class SoftJaccardLossWithLogits(SoftJaccardLoss):
    def forward(self, prediction, target):
        return super().forward(torch.sigmoid(prediction), target)


class dice_wbce(nn.Module):
    def __init__(self, positive_ratio=0.15, alpha=0.5):
        super().__init__()
        self.a=alpha
        self.criterion_1 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([(1-positive_ratio)/positive_ratio]))
        self.criterion_2 = DiceLoss(sigmoid=True)

    def forward(self, logits, target):
        c1 = self.criterion_1(logits, target)
        c2 = self.criterion_2(logits, target)
        return (self.a*c1) + ((1-self.a)*c2)

class logdice_wbce(nn.Module):
    def __init__(self, positive_ratio=0.15, alpha=0.5):
        super().__init__()
        self.a=alpha
        self.criterion_1 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([(1-positive_ratio)/positive_ratio]))
        self.criterion_2 = SoftDiceLossWithLogits()

    def forward(self, logits, target):
        c1 = self.criterion_1(logits, target)
        c2 = - torch.log(1-self.criterion_2(logits, target))
        return (self.a*c1) + ((1-self.a)*c2)


class jaccard_wbce(nn.Module):
    def __init__(self, positive_ratio=0.15, alpha=0.5):
        super().__init__()
        self.a=alpha
        self.criterion_1 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([(1-positive_ratio)/positive_ratio]))
        self.criterion_2 = SoftJaccardLossWithLogits()

    def forward(self, logits, target):
        c1 = self.criterion_1(logits, target)
        c2 = self.criterion_2(logits, target)
        return (self.a*c1) + ((1-self.a)*c2)

class logjaccard_wbce(nn.Module):
    def __init__(self, positive_ratio=0.15, alpha=0.5):
        super().__init__()
        self.a=alpha
        self.criterion_1 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([(1-positive_ratio)/positive_ratio]))
        self.criterion_2 = SoftJaccardLossWithLogits()

    def forward(self, logits, target):
        c1 = self.criterion_1(logits, target)
        c2 = - torch.log(1-self.criterion_2(logits, target))
        return (self.a*c1) + ((1-self.a)*c2)