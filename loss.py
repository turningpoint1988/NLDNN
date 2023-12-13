#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criteria = nn.BCELoss()

    def forward(self, prediction, target):
        prediction = prediction.view(-1)
        target = target.view(-1)
        loss = self.criteria(prediction, target)
        return loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criteria = nn.MSELoss()

    def forward(self, prediction, target):
        prediction = prediction.view(-1)
        target = target.view(-1)
        loss = self.criteria(prediction, target)
        return loss
        
        
class PoissonLoss(nn.Module): 
    def __init__(self):
        super(PoissonLoss, self).__init__()
        self.criteria = nn.PoissonNLLLoss(log_input=True)

    def forward(self, prediction, target):
        prediction = prediction.view(-1)
        target = target.view(-1)
        loss = self.criteria(prediction, target)
        return loss
