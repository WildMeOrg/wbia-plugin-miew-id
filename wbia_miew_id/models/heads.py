import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)wandb: ERROR Abnormal program exit

        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
    

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output
    
class ElasticArcFace(nn.Module):

    def __init__(self, loss_fn, in_features, out_features, s=64.0, m=0.50,std=0.0125,plus=False, k=None):
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std = std
        self.plus = plus
        self.loss_fn = loss_fn

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device) # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
    
        loss = self.loss_fn(cos_theta, label)

        return loss
    
########## Subcenter Arcface with dynamic margin ##########

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine
    
class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, out_dim, s):
        super().__init__()
        self.s = s
        self.register_buffer('margins', torch.tensor(margins))
        self.out_dim = out_dim
            
    def forward(self, logits, labels):
        #ms = []
        #ms = self.margins[labels.cpu().numpy()]
        ms = self.margins[labels]
        cos_m = torch.cos(ms) #torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.sin(ms) #torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.cos(math.pi - ms)#torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.sin(math.pi - ms) * ms#torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        cosine = logits
        sine = torch.sqrt(1.0 - cosine * cosine)
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        return output
    
class ArcFaceSubCenterDynamic(nn.Module):
    def __init__(
        self,
        loss_fn,
        embedding_dim,
        output_classes,
        margins,
        s,
        k=2,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_classes = output_classes
        self.margins = margins
        self.s = s
        self.wmetric_classify = ArcMarginProduct_subcenter(self.embedding_dim, self.output_classes, k=k)
        
        self.warcface_margin = ArcFaceLossAdaptiveMargin(margins=self.margins, 
                                                         out_dim=self.output_classes, 
                                                         s=self.s)

        self.loss_fn = loss_fn

    def forward(self, features, labels):
        logits = self.wmetric_classify(features.float())
        logits = self.warcface_margin(logits, labels)
        loss = self.loss_fn(logits, labels)
        return loss