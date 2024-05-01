import math
import numpy as np
import torch
import torch.nn as nn
import logging
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from transformers import AutoConfig
from transformers import AutoModelWithLMHead

from utils import *
from dataloader import *

logger = logging.getLogger()

class BertTagger(nn.Module):

    def __init__(self, output_dim, params):
        super(BertTagger, self).__init__()
        self.hidden_dim = params.hidden_dim
        self.output_dim = output_dim
        config = AutoConfig.from_pretrained(params.model_name)
        config.output_hidden_states = True
        config.output_attentions = True
        self.encoder = AutoModelWithLMHead.from_pretrained(params.model_name, config=config)
        # if params.ckpt: # False
        #     logger.info("Reloading encoder from %s" % params.ckpt)
        #     encoder_ckpt = torch.load(params.ckpt)
        #     self.encoder.load_state_dict(encoder_ckpt)
        # 分类器
        self.classifier = CosineLinear(self.hidden_dim, self.output_dim)


    def forward(self, X, return_feat=False):
        features = self.forward_encoder(X)
        logits,_,_ = self.forward_classifier(features)
        if return_feat:
            return logits, features
        return logits
        
    def forward_encoder(self, X):
        features = self.encoder(X)
        features = features[1][-1]
        return features

    def forward_classifier(self, features):
        logits = self.classifier(features)
        return logits 



class CosineLinear(nn.Module):
    def __init__(self, hidden_dim, output_dim, sigma=True):
        super(CosineLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.Tensor(output_dim, hidden_dim))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input, num_head=1):

        out = F.linear(F.normalize(input, p=2,dim=1), \
            F.normalize(self.weight, p=2, dim=1))  # （bs, seq_len, hidden_dim） w: (out_dim, hidden_dim)


        if self.sigma is not None:
            out = self.sigma * out

        return out

class SplitCosineLinear(nn.Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, hidden_dim, old_output_dim, new_output_dim, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = old_output_dim + new_output_dim
        self.fc0 = CosineLinear(hidden_dim, 1, False)
        self.fc1 = CosineLinear(hidden_dim, old_output_dim-1, False)
        self.fc2 = CosineLinear(hidden_dim, new_output_dim, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x, num_head=1):
        out0 = self.fc0(x, num_head=num_head)
        out1 = self.fc1(x, num_head=num_head)
        out2 = self.fc2(x, num_head=num_head)

        out = torch.cat((out0, out1, out2), dim=-1)
       
        if self.sigma is not None:
            out = self.sigma * out
        return out

