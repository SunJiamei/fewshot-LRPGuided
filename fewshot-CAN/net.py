import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchFewShot.LRPtools import utils as LRPutil
from .resnet12 import resnet12
from .cam import CAM


class Model(nn.Module):
    def __init__(self, scale_cls, num_classes=64):
        super(Model, self).__init__()
        self.scale_cls = scale_cls
        self.base = resnet12()
        self.cam = CAM()
        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 

    def test(self, ftrain, ftest):
        ftest = ftest.mean(4)
        ftest = ftest.mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def extract_feature(self, x):
        x = x.unsqueeze(0)
        f = self.base(x)
        return f

    def forward(self, xtrain, xtest, ytrain, ytest):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)
        f_h = f.size(-2)
        f_w = f.size(-1)
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1)
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])
        ftrain, ftest, _, _ = self.cam(ftrain, ftest)
        ftrain = ftrain.mean(4)
        ftrain = ftrain.mean(4)

        if not self.training:
            return self.test(ftrain, ftest)

        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])

        ftest = ftest.view(batch_size, num_test, K, -1)
        ftest = ftest.transpose(2, 3)
        ytest = ytest.unsqueeze(3)
        ftest = torch.matmul(ftest, ytest)
        ftest = ftest.view(batch_size * num_test, -1, f_h, f_w)
        ytest = self.clasifier(ftest)

        return ytest, cls_scores



'''LRP weighted'''

class ModelwithLRP(nn.Module):
    def __init__(self, scale_cls, num_classes=64):
        super(ModelwithLRP, self).__init__()
        self.scale_cls = scale_cls

        self.base = resnet12()
        self.cam = CAM()

        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)

    def test(self, ftrain, ftest):
        ftest = ftest.mean(-1)
        ftest = ftest.mean(-1)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)

        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def forward(self, xtrain, xtest, ytrain, ytest):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)

        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1)
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])
        ftrain, ftest, _, _ = self.cam(ftrain, ftest)


        ftest_att = ftest
        ftrain = ftrain.mean(-1)
        ftrain = ftrain.mean(-1)
        ftrain_mean2 = ftrain

        if not self.training:
            return self.test(ftrain, ftest)
        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)

        ftest_mean1 = ftest_att.mean(-1)
        # print(ftest.sum(), ftest_mean1.sum())
        ftest_mean2 = ftest_mean1.mean(-1)  # (b, n2, n1, c)
        # print(ftest_mean2.sum())
        ftest_mean2_norm = F.normalize(ftest_mean2, p=2, dim=ftest_mean2.dim() - 1,
                                       eps=1e-12)  # this is the attended test features, each test sample corresponds to a set of features
        ftrain_mean2_norm = F.normalize(ftrain_mean2, p=2, dim=ftrain_mean2.dim() - 1,
                                        eps=1e-12)  # this is the attended centroid, each test sample corresponds to a set of centroids

        scores = self.scale_cls * torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1)  #(b, n2, n1)

        # calculate the logits from the cosine similarity
        logits_sf = torch.softmax(scores, dim=-1)
        # initialize the relevance of each label
        relevance_logits = torch.log(LRPutil.LOGIT_BETA * (logits_sf +LRPutil.EPSILON)/ (torch.tensor([1 + LRPutil.EPSILON]).cuda() - logits_sf))

        relevance_ftest_mul_ftrain = LRPutil.compute_lrp_sum(
            torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1),
            ftest_mean2_norm * ftrain_mean2_norm,
            relevance_logits, dim=-1)  #(b, n2, n1, c)
        relevance_weight = LRPutil.normalize_relevance(relevance_ftest_mul_ftrain.squeeze(-1))

        relevance_ftest_mean1 = LRPutil.compute_lrp_mean(ftest_mean2, ftest_mean1, relevance_ftest_mul_ftrain, dim=-1)
        relevance_ftest = LRPutil.compute_lrp_mean(ftest_mean1, ftest_att, relevance_ftest_mean1, dim=-1)  #(b, n2, n1, c, h, w)

        relevance_ftest = LRPutil.normalize_relevance(relevance_ftest,dim=3)
        # for the CAN model we only use the lrp weighted features to train.
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm * relevance_ftest * relevance_weight.unsqueeze(-1).unsqueeze(-1), dim=3)
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])


        ftest = ftest.view(batch_size, num_test, K, -1)
        ftest = ftest.transpose(2, 3)
        ytest = ytest.unsqueeze(3)
        ftest = torch.matmul(ftest, ytest)
        ftest = ftest.view(batch_size * num_test, -1, 6, 6)
        ytest = self.clasifier(ftest)
        return ytest, cls_scores

