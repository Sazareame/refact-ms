import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
import numpy as np
# import torch.nn.functional as F
import math
from typing import Optional, Any, Union, Callable
from mindspore import Tensor
from mindspore.common.initializer import Normal, Constant, initializer


def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()


class LinearBlock(nn.Cell):
    def __init__(self, in_features, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Dense(in_features, out_features)
        self.fc.weight.set_data(
            initializer(Normal(math.sqrt(2. / out_features), 0)))

    def construct(self, x):
        x = self.fc(x)
        return x

def normalize_digraph(A: mindspore.Tensor):
    b, n, _ = A.shape  # (bs, num_classes, num_classes)
    # node_degrees = A.detach().sum(dim = -1)
    node_degrees = ops.stop_gradient(A).sum(axis=-1)
    degs_inv_sqrt = node_degrees ** -0.5
    norm_degs_matrix = ops.eye(n)
    norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1)
    norm_A = ops.bmm(ops.bmm(norm_degs_matrix,A),norm_degs_matrix)
    return norm_A

class CrossAttn(nn.Module):
    """ cross attention Module"""
    def __init__(self, in_channels):
        super(CrossAttn, self).__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** -0.5
        self.attend = nn.Softmax(dim=-2)

        self.linear_k.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_q.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2. / in_channels))

    def forward(self, y, x):
        query = self.linear_q(y)  # (bs, 12, h*w, c//2)  (bs, 12, c//2)
        key = self.linear_k(x)  # (bs, 12, h*w, c//2)  (bs, 1, c//2)
        value = self.linear_v(x)  # (bs, 12, h*w, c)  (bs, 1, c)
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (bs, 12, h*w, h*w) (bs, 12, 1)  
        attn = self.attend(dots)
        out = torch.matmul(attn, value)  # (bs, 12, h*w, h*w)*(bs, 12, h*w, c)->(bs, 12, h*w, c)  # (bs, 12, 1)*(bs, 1, c)->(bs, 12, c)
        return out

class GNN(nn.Cel):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Dense(self.in_channels,
                          self.in_channels,
                          weight_init=Normal(math.sqrt(2. / self.in_channels)))
        self.V = nn.Dense(self.in_channels,
                          self.in_channels,
                          weight_init=Normal(math.sqrt(2. / self.in_channels)))
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        # self.bnv.weight.data.fill_(1)
        # self.bnv.bias.data.zero_()
        self.bnv.gamma.set_data(initializer(Constant(1)))
        self.bnv.beta.set_data(initializer(Constant(0)))
        self.l2_normalize = ops.L2Normalize(axis=-1)()

    def construct(self, x):
        b, n, c = x.shape

        # build dynamical graph
        if self.metric == 'dots':
            # si = x.detach()  # (bs, num_classes, c')
            si = ops.stop_gradient(x)
            si = ops.einsum('b i j , b j k -> b i k', si, si.swapaxes(1, 2))  # (bs, num_classes, num_classes)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)  # (bs, num_classes, 1)
            adj = (si >= threshold).float()  # (bs, num_classes, num_classes)

        elif self.metric == 'cosine':
            # si = x.detach()
            si = ops.stop_gradient(x)
            # si = F.normalize(si, p=2, dim=-1)
            si = self.l2_normalize(si)
            si = ops.einsum('b i j , b j k -> b i k', si, si.swapaxes(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'l1':
            # si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = ops.stop_gradient(x).tile((1, n, 1)).view(b, n, n, c)
            si = ops.abs(si.swapaxes(1, 2) - si)
            si = si.sum(axis=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        else:
            raise Exception("Error: wrong metric: ", self.metric)

        # GNN process
        A = normalize_digraph(adj)
        aggregate = ops.einsum('b i j, b j k->b i k', A, self.V(x))
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return x

class GEM(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GEM, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.FAM = CrossAttn(self.in_channels)
        self.ARM = CrossAttn(self.in_channels)
        self.edge_proj = nn.Linear(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(self.num_classes * self.num_classes)

        self.edge_proj.weight.data.normal_(0, math.sqrt(2. / in_channels))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, class_feature, global_feature):
        B, N, D, C = class_feature.shape  # (bs, 12, h*w, c)
        global_feature = global_feature.repeat(1, N, 1).view(B, N, D, C)  # (bs, h*w, c)->(bs, 12, h*w, c)
        feat = self.FAM(class_feature, global_feature)  # (bs, 12, h*w, c)
        feat_end = feat.repeat(1, 1, N, 1).view(B, -1, D, C)
        feat_start = feat.repeat(1, N, 1, 1).view(B, -1, D, C)
        feat = self.ARM(feat_start, feat_end)
        edge = self.bn(self.edge_proj(feat))
        return edge
    
def create_e_matrix(n):
    end = torch.zeros((n*n,n))
    for i in range(n):
        end[i * n:(i + 1) * n, i] = 1
    start = torch.zeros(n, n)
    for i in range(n):
        start[i, i] = 1
    start = start.repeat(n,1)
    return start,end

class MAD(nn.Cell):
    def __init__(self, local_attention_num=12, p=0.0):
        super(MAD, self).__init__()
        self.local_attention_num = local_attention_num
        self.p = p

    def construct(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]
        
        if self.p == 0.0:
            return inputs

        bs = inputs.shape[0]
        index = ops.randint(0, self.local_attention_num, (bs,))
        
        for i in range(bs):
            if ops.rand(1,) > (1.0 - self.p):
                # inputs[0, index[0], ...] = 0
                mask = ops.ones_like(inputs)
                mask[i, index[i], ...] = 0
                outs = inputs * mask
                # inputs[i, index[i], ...] = 0 
             
        # print(inputs[0, index[0], ...])
        return outs

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class MyTransformerEncoderLayer(nn.Module):
   
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 3072, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MyTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)

        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)



        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)


        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)


        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(MyTransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
       

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class Residual_Attention_Neck_Official(nn.Cell):
    """Class-specific residual attention classifier head.

    Residual Attention: A Simple but Effective Method for Multi-Label
                        Recognition (ICCV 2021)
    Please refer to the `paper <https://arxiv.org/abs/2108.02456>`__ for
    details.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        num_heads (int): Number of residual at tensor heads.
        loss (dict): Config of classification loss.
        lam (float): Lambda that combines global average and max pooling
            scores.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """
    temperature_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_heads,
                 lam,
                 use_gap=True):
        assert num_heads in self.temperature_settings.keys(
        ), 'The num of heads is not in temperature setting.'
        assert lam > 0, 'Lambda should be between 0 and 1.'
        super(Residual_Attention_Neck_Official, self).__init__()
        self.temp_list = self.temperature_settings[num_heads]
        self.csra_heads = nn.CellList([
            CSRAModule(num_classes, in_channels, self.temp_list[i], lam, use_gap)
            for i in range(num_heads)
        ])


    def construct(self, x):
        logit = 0.
        for head in self.csra_heads:
            logit += head(x)
        return logit

class CSRAModule(nn.Cell):
    """Basic module of CSRA with different temperature.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        T (int): Temperature setting.
        lam (float): Lambda that combines global average and max pooling
            scores.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self, num_classes, in_channels, T, lam, use_gap=True, init_cfg=None):

        super(CSRAModule, self).__init__(init_cfg=init_cfg)
        self.T = T  # temperature
        self.lam = lam  # Lambda
        self.head = nn.Conv2d(in_channels, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.use_gap = use_gap

    def forward(self, x):
        score = self.head(x) / torch.norm(
            self.head.weight, dim=1, keepdim=True).transpose(0, 1)
        score = score.flatten(2) # (bs, numcls, h*w)

        if self.use_gap:
            base_logit = torch.mean(score, dim=2)
        else:
            base_logit = 0.0

        if self.T == 99:  # max-pooling
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T) # (bs, numcls, h*w)
            att_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.lam * att_logit

class Anfl_Neck(nn.Cell):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots', 
                 local_attention_num=12, channel_reduction_rate=2, use_lanet=False, p=0.0, 
                 use_csra=False, lam=1.0, use_gap=True, num_heads=8, 
                 with_backbone_logit=False, 
                 use_self_attn_fusion=False, encoder_layers_num=6, multi_heads_num=8):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.CellList(class_linear_layers)

        self.gnn = GNN(self.in_channels,
                       self.num_classes,
                       neighbor_num=neighbor_num,
                       metric=metric)
        
        # self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        
        self.relu = nn.ReLU()
        self.global_linear = LinearBlock(self.in_channels, self.in_channels)

        self.local_attention_num = local_attention_num
        self.use_lanet = use_lanet

        self.lam = lam
        self.use_gap = use_gap
        self.num_heads = num_heads
        self.use_csra = use_csra
        self.with_backbone_logit = with_backbone_logit
        
        self.use_self_attn_fusion = use_self_attn_fusion
        self.encoder_layers_num = encoder_layers_num 
        self.multi_heads_num = multi_heads_num

        assert not(self.use_csra and self.with_backbone_logit)

        if use_csra:
            self.residual_attention = Residual_Attention_Neck_Official(num_classes, in_channels, 
                                                                num_heads, lam, use_gap)
        if use_lanet:
            self.lanets = nn.SequentialCell()
            for i in range(local_attention_num):
                self.lanets.insert_child_to_cell('neck_lanets{}'.format(i), 
                                    nn.SequentialCell(nn.Conv2d(in_channels, in_channels//channel_reduction_rate, 1),
                                                #   nn.ReLU(),
                                                  nn.Conv2d(in_channels//channel_reduction_rate, 1, 1)))
        # nn.init.xavier_uniform_(self.sc)
        self.mad = MAD(num_classes, p)


        # self.edge_extractor = GEM(self.in_channels, self.num_classes)
        
        if use_self_attn_fusion:
            self.fusion_weight_token = mindspore.Parameter(ops.zeros(1, 1, in_channels))
            self.encoder_layers = nn.Sequential()
            for i in range(encoder_layers_num):
                self.encoder_layers.add_module('neck_transformer_layer{}'.format(i), MyTransformerEncoderLayer(d_model=in_channels, 
                nhead=multi_heads_num, batch_first=True))

    def forward(self, x):
        
        if self.with_backbone_logit:
            x_ = x
            backbone_out = x_[1]
            x = x_[0]
            
        if self.use_csra:
            csra_outs = self.residual_attention(x)

        # local_cnn
        if self.use_lanet:
            lanet_outs = [] # x (bs, c, h, w)
            for i in range(self.local_attention_num):
                lanet_outs.append(self.lanets[i](x))
            lanet_outs = tuple(lanet_outs)
            
            outs = torch.sigmoid(torch.cat(lanet_outs, dim=1)) # (bs, atten_map_num, h, w)
            # outs = torch.sigmoid(outs)
            
            outs = self.mad(outs)
            local_atten_map = torch.max(outs, 1, keepdim=True)[0]
            x = torch.mul(local_atten_map, x) #(b, 512, h, w)
        

        b, c, _, _ = x.shape
        x = x.view(b,c,-1).permute(0,2,1)

        x = self.global_linear(x)

        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))  # (bs, h*w, c) -> (bs, 1, h*w, c')
        f_u = torch.cat(f_u, dim=1)  # (bs, num_classes, h*w, c')

        f_v = f_u.mean(dim=-2)  # gap (bs, num_classes, c')

        # FGG
        f_v = self.gnn(f_v)  # (bs, num_classes, c')

        # f_e = self.edge_extractor(f_u, x)
        # f_e = f_e.mean(dim=-2)
        # f_v, f_e = self.gnn(f_v, f_e)

        # b, n, c = f_v.shape
        # sc = self.sc
        # sc = self.relu(sc)
        # sc = F.normalize(sc, p=2, dim=-1)
        # cl = F.normalize(f_v, p=2, dim=-1)

        # cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        # return cl
        if self.use_self_attn_fusion:
            fusion_weight_token = self.fusion_weight_token.expand(b, -1, -1)
            _backbone_out = backbone_out.unsqueeze(1)
            trans_inputs = torch.cat((fusion_weight_token, _backbone_out, f_v), dim=1)  # [B, 1 + 1 + 12, in_channel]
            trans_outs = self.encoder_layers(trans_inputs)[:, 0]
            
            # fusion_logits = self.bn_fc(self.cross_attn_fusion(f_v, backbone_out)).permute(0, 2, 1).contiguous()
            # fusion_logits = self.bn(fusion_logits).permute(0, 2, 1).contiguous()
            
        
        
        if self.with_backbone_logit and not self.use_self_attn_fusion:
            return f_v, backbone_out
        elif self.use_csra:
            return f_v, csra_outs
        elif self.use_self_attn_fusion:
            return f_v, backbone_out, trans_outs
        else:
            return f_v

    def simple_test(self, x):

        if self.with_backbone_logit:
            x_ = x
            backbone_out = x_[1]
            x = x_[0]

        if self.use_csra:
            csra_outs = self.residual_attention(x)

        # local_cnn
        if self.use_lanet:
            lanet_outs = [] # x (bs, c, h, w)
            for i in range(self.local_attention_num):
                lanet_outs.append(self.lanets[i](x))
            lanet_outs = tuple(lanet_outs)
            
            outs = torch.sigmoid(torch.cat(lanet_outs, dim=1)) # (bs, atten_map_num, h, w)
            local_atten_map = torch.max(outs, 1, keepdim=True)[0]
            x = torch.mul(local_atten_map, x) #(b, 512, h, w)

        b, c, _, _ = x.shape
        x = x.view(b,c,-1).permute(0,2,1)

        x = self.global_linear(x)

        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))  # (bs, h*w, c) -> (bs, 1, h*w, c')
        f_u = torch.cat(f_u, dim=1)  # (bs, num_classes, h*w, c')
        f_v = f_u.mean(dim=-2)  # gap (bs, num_classes, c')
        # FGG
        f_v = self.gnn(f_v)  # (bs, num_classes, c')

        # f_e = self.edge_extractor(f_u, x)
        # f_e = f_e.mean(dim=-2)
        # f_v, f_e = self.gnn(f_v, f_e)

        
        if self.use_self_attn_fusion:
            fusion_weight_token = self.fusion_weight_token.expand(b, -1, -1)
            _backbone_out = backbone_out.unsqueeze(1)
            trans_inputs = torch.cat((fusion_weight_token, _backbone_out, f_v), dim=1)  # [B, 1 + 1 + 12, in_channel]
            trans_outs = self.encoder_layers(trans_inputs)[:, 0]

        
        if self.with_backbone_logit and not self.use_self_attn_fusion:
            return f_v, backbone_out
        elif self.use_csra:
            return f_v, csra_outs
        elif self.use_self_attn_fusion:
            return f_v, backbone_out, trans_outs
        else:
            return f_v



