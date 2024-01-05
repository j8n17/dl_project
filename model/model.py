import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class TSN(BaseModel):
    def __init__(self, num_segments, modality='rgb', base_model='resnet50', consensus_type='avg', partial_bn=True, num_classes=2):
        super().__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.num_classes = num_classes
        self.consensus_type = consensus_type

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_classes)
        self.consensus = ConsensusModule(consensus_type)
        self._enable_pbn = partial_bn
    
    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        else:
            print('model is not ready yet.')

    def _prepare_tsn(self, num_classes):
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, input):
        # input의 shape : (Batch, Num_segments, RGB, Height, Weight)
        # output의 shape : (Batch, Num_classes)

        if self.modality != 'rgb':
            print('This modality is not ready yet.')
            raise AttributeError
        else:
            # (Batch, Num_seg, 3, H, W)
            input = input.view((-1, 3) + input.size()[-2:]) # input의 shape : (Batch*Num_seg, RGB, H, W)
            # 기존 TSN에는 sample_len이라는 변수 존재
            base_out = self.base_model(input) # base_out의 shape : (Batch*Num_seg, Num_classes)
            base_out = base_out.view((-1, self.num_segments) + (base_out.size()[-1],)) # (Batch, Num_seg, Num_classes)

            output = self.consensus(base_out) # (Batch, Num_classes)
            return output
        
    def get_optim_policies(self):
        # BN Freezing 또는 모듈별 lr 설정을 위해 Optimizer에게 줄 파라미터를 조정하는 함수
        # output은 Optimizer에 입력할 수 있는 파라미터
        first_conv_weight = []
        first_conv_bias = []
        remainder_weight = []
        remainder_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    remainder_weight.append(ps[0])
                    if len(ps) == 2:
                        remainder_bias.append(ps[1])
            elif isinstance(m, nn.Linear):
                ps = list(m.parameters())
                remainder_weight.append(ps[0])
                if len(ps) == 2:
                    remainder_bias.append(ps[1])
                  
            elif isinstance(m, nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': remainder_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "remainder_weight"},
            {'params': remainder_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "remainder_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]


class ConsensusModule(nn.Module):
    def __init__(self, consensus_type):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type

    def forward(self, input):
        self.shape = input.size() # (Batch, Num_seg, Num_classes)
        if self.consensus_type == 'avg':
            output = input.mean(dim=1)

        return output