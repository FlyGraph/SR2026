import torch
import torch.nn as nn
from core.base_network import BaseNetwork

class VBaselineNetwork(BaseNetwork):
    def __init__(self, init_type='kaiming', gain=0.02, z_times=6, **kwargs):
        super(VBaselineNetwork, self).__init__(init_type=init_type, gain=gain)
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.z_times = z_times

    def init_weights(self, init_type='normal', gain=0.02):
        pass

    def set_loss(self, loss_fn):
        pass

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        pass

    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, adjust=False, path=None):
        
        
        img_below = y_cond[:, 0:1, :, :] 
        img_up = y_cond[:, 1:2, :, :]    
        
        
        outputs = []
        for i in range(1, self.z_times):
            weight = i / float(self.z_times)
            interpolated = (1 - weight) * img_up + weight * img_below
            outputs.append(interpolated)
            
        output = torch.cat(outputs, dim=1)
        
        return output, [output]

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        return torch.tensor(0.0, device=y_0.device, requires_grad=True)
