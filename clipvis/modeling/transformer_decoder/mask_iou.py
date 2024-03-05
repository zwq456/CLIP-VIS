import torch
from torch import nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init

class MaskIoUFeatureExtractor(nn.Module):
    """
    MaskIou head feature extractor.
    """

    def __init__(self, input_channels,hidden_dim,out_dim):
        super(MaskIoUFeatureExtractor, self).__init__()
        
        # input_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM+cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # hidden_dim=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.maskiou_fcn1 = torch.nn.Conv2d(input_channels, hidden_dim, 3, 1, 1) 
        self.maskiou_fcn2 = torch.nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1) 
        self.maskiou_fcn3 = torch.nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1) 
        self.maskiou_fcn4 = torch.nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1) 

        self.maskiou_fc1 = nn.Linear(hidden_dim*16*16, 1024)
        self.maskiou_fc2 = nn.Linear(1024, 1024)
        self.maskiou = nn.Linear(1024, out_dim)

        for l in [self.maskiou_fcn1, self.maskiou_fcn2, self.maskiou_fcn3,self.maskiou_fcn4]:
            # nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            # nn.init.constant_(l.bias, 0)
            weight_init.c2_xavier_fill(l)

    
    def forward(self, x, mask):
        b,q=x.shape[:2]
        x=x.flatten(0,1)
        mask=mask.flatten(0,1)
        x = torch.cat((x, mask), 1)
        x = F.relu(self.maskiou_fcn1(x))
        x = F.relu(self.maskiou_fcn2(x))
        x = F.relu(self.maskiou_fcn3(x))
        x = F.relu(self.maskiou_fcn4(x))
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))
        maskiou = self.maskiou(x)
        # maskiou=torch.sigmoid(maskiou)
        maskiou=maskiou.reshape(b,q,-1)
        # maskiou=torch.split(maskiou,split_size_or_sections=2,dim=0)
        # maskiou=torch.stack(maskiou,dim=1)
        return maskiou
        