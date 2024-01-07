import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Constant, Normal, initializer

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     pad_mode='pad',
                     padding=dilation,
                     group=groups,
                     dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     stride=1)

class IBssicBlock(nn.Cell):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 with_cp=False):
        super().__init__()
        self,bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prlue = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.with_cp = with_cp
    
    def construct(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out
    

class IResNwt(nn.Cell):
    fc_scale = 7 * 7
    def __init__(self,
                 layers,
                 block=IBssicBlock,
                 dropout=0,
                 num_features=512,
                 with_pooling_fc=True,
                 with_pooling_fc_out=False,
                 use_anfl=False,
                 with_cp=False):
        super().__init__()

        self.inplanes = 64
        self.dilation = 1
        self.with_cp = with_cp
        self.with_pooling_fc = with_pooling_fc
        self.with_pooling_fc_out = with_pooling_fc_out
        self.use_anfl = use_anfl

        assert not (self.with_pooling_fc and self.with_pooling_fc_out)

        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            pad_mode='pad',
            padding=1,
            has_bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block,
                                       64,
                                       layers[0],
                                       stride=2,
                                       with_cp=self.with_cp)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       with_cp=self.with_cp
                                       )
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       with_cp=self.with_cp
                                       )
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       with_cp=self.with_cp
                                       )
        self.bn2 = nn.BatchNorm2d(512 * block.expansion)

        if self.with_pooling_fc or self.with_pooling_fc_out:
            self.dropout = nn.Dropout(p=dropout) # to impl: inplace=True
            self.fc = nn.Dense(512 * block.expansion * self.fc_scale,
                               num_features) 
            self.features = nn.BatchNorm1d(num_features)
        # Here in pytorch version, nn.init.constant_() is applied to
        # self.features. I could not find the corresponding function
        # in mindspore and do not know the reason either.
        
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(initializer(Normal(0.1, 0)))
            # Also here the weight and bias of BatchNorm2d are initialized
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, with_cp=False):
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1(self.inplanes,
                        planes * block.expansion,
                        stride),
                nn.BatchNorm2d(planes * block.expansion) 
            )
        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, with_cp=with_cp
        ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes
            ))
        return nn.SequentialCell(*layers)
    
    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        if self.with_pooling_fc:
            x = ops.flatten(x) # might raise error since I do not exactly know the difference between two APIs
            x = self.dropout(x)
            x = self.fc(x)
            x = self.features(x)
        elif self.with_pooling_fc_out:
            x_ = x
            x_ = ops.flatten(x_) # might raise error since I do not exactly know the difference between two APIs
            x_ = self.dropout(x_)
            x_ = self.fc(x_)
            x_ = self.features(x_)
        
        # the operation to be done if set `use_anfl` was commented in pytorch version code, WHY?
        # 破案了, this is because anfl is implemented in neck, correct?
        return x, x_ if self.with_pooling_fc_out else x