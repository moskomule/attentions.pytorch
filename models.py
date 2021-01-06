import copy
from functools import partial
from typing import Callable, Iterator, Optional

import torch
from torch import nn
from torch.nn import functional as F

has_opt_einsum = False
try:
    import opt_einsum

    has_opt_einsum = True
except ImportError:
    print("no opt_einsum")


def einsum(expr: str,
           *xs):
    if has_opt_einsum:
        return opt_einsum.contract(expr, *xs, backend='torch')
    return torch.einsum(expr, *xs)


class EMA(nn.Module):
    def __init__(self,
                 original_model: nn.Module,
                 beta: float = 0.999):
        super().__init__()
        self._original_model = original_model
        self._ema_model = copy.deepcopy(original_model)
        for p in self._ema_model.parameters():
            p.requires_grad_(False)
        self.beta = beta

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        # this makes it simple, but may incur unexpected behavior
        return self._original_model.parameters(recurse)

    def requires_grad_(self, requires_grad: bool = True):
        return self._original_model.requires_grad_(requires_grad)

    @torch.no_grad()
    def _update(self):
        o_p = [p.data for p in self._original_model.parameters() if torch.is_tensor(p)]
        e_p = [p.data for p in self._ema_model.parameters() if torch.is_tensor(p)]
        torch._foreach_mul_(e_p, self.beta)
        torch._foreach_add_(e_p, o_p, alpha=1 - self.beta)

        o_b = [b for b in self._original_model.buffers() if torch.is_tensor(b) and torch.is_floating_point(b)]
        e_b = [b for b in self._ema_model.buffers() if torch.is_tensor(b) and torch.is_floating_point(b)]
        torch._foreach_mul_(e_b, self.beta)
        torch._foreach_add_(e_b, o_b, alpha=1 - self.beta)

    def forward(self, *args, **kwargs):
        if self.training:
            self._update()
            return self._original_model(*args, **kwargs)
        return self._ema_model(*args, **kwargs)

    def __repr__(self):
        s = f"EMA(beta={self.beta},\n"
        s += f"  {self._original_model}\n"
        s += ")"
        return s


def conv1x1(in_channels: int,
            out_channels: int,
            bias: bool):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)


class SelfAttention(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dim_key: int = 64,
                 num_heads: int = 1,
                 efficient_attention: bool = False,
                 lambda_attention: bool = False,
                 bias: bool = False,
                 scale_factor: float = None,
                 kv_reduction: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
                 ):
        """

        Args:
            in_channels: input channels
            out_channels: output channels
            dim_key: dimension of keys an queries
            num_heads: number of heads
            efficient_attention: if use global attention
            bias: if use bias for convolutions
            scale_factor: scale factor that is multiplied to query and key (default: key_dim ** -0.25)
            kv_reduction: Spatial pooling function applied to key and values
        """

        super().__init__()
        assert not (efficient_attention and lambda_attention)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_qheads = num_heads
        self.num_kvheads = 1 if lambda_attention else num_heads
        self.dim_query = dim_key * num_heads if lambda_attention else dim_key
        self.dim_key = dim_key
        self.dim_value = out_channels // num_heads if lambda_attention else out_channels
        self.efficient_attention = efficient_attention
        self.lambda_attention = lambda_attention
        self.scale_factor = scale_factor or self.dim_key ** -0.25
        self.kv_reduction = kv_reduction or nn.Identity()

        self.query = nn.Sequential(conv1x1(in_channels, self.dim_query, False),
                                   nn.BatchNorm2d(self.dim_query, momentum=1 - 0.9999))
        self.key = nn.Sequential(conv1x1(in_channels, self.dim_key, bias),
                                 nn.BatchNorm2d(self.dim_key, momentum=1 - 0.9999))
        self.value = conv1x1(in_channels, self.dim_value, bias)
        if self.lambda_attention:
            self.register_parameter("rpe", None)
            self.register_buffer("pos", None)

    def _init_rpe(self, size):
        # relative position embedding, from https://github.com/lucidrains/lambda-networks
        r = 2 * size - 1
        self.register_parameter("rpe", nn.Parameter(torch.randn(r, r, self.dim_key)))
        pos = torch.arange(size)
        pos = torch.stack(torch.meshgrid(pos, pos)).permute(1, 2, 0).flatten(0, 1)  # [n*n,2]; pos[n] = (i, j)
        rel_pos = pos[None, :] - pos[:, None]  # [n*n, n*n, 2]; rel_pos[n, m] = rel(i, j)
        rel_pos += size - 1
        self.register_buffer("pos", rel_pos)

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:

        q = self.query(input).flatten(-2)
        k = self.kv_reduction(self.key(input)).flatten(-2)
        v = self.kv_reduction(self.value(input)).flatten(-2)

        b = input.size(0)
        q = q.reshape(b, self.num_qheads, self.dim_query // self.num_qheads, -1) * self.scale_factor
        k = k.reshape(b, self.num_kvheads, self.dim_key // self.num_kvheads, -1) * self.scale_factor
        v = v.reshape(b, self.num_kvheads, self.dim_value // self.num_kvheads, -1)

        if self.efficient_attention:
            # See Shen+2021 Efficient Attention: Attention with Linear Complexities for more details
            q = q.softmax(dim=-2)
            context = einsum("bhkn,bhvn->bhkv", k.softmax(dim=-1), v)
            out = einsum("bhkv,bhkn->bhvn", context, q)

        elif self.lambda_attention:
            # See Anonymous 2021 LambdaNetworks: Modeling long-range Interactions without Attention for mode details
            k.squeeze_(1)
            v.squeeze_(1)
            context = einsum("bkn,bvn->bkv", k.softmax(dim=-1), v)
            out = einsum("bkv,bhkn->bhvn", context, q)
            if self.rpe is None:
                self._init_rpe(max(input.shape[-2:]))
            n, m = self.pos.unbind(dim=-1)
            rpe = self.rpe[n, m]
            pos_context = einsum("nmk,bvm->bnkv", rpe, v)
            out += einsum("bhkn,bnkv->bhvn", q, pos_context)

        else:
            # dot product attention
            context = einsum("bhkn,bhkm->bhmn", q, k).softmax(dim=-1)
            out = einsum("bhmn,bhvm->bhvn", context, v)
        return out.reshape(b, -1, input.size(-2), input.size(-1))

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, key_dim={self.dim_key} " \
               f"value_dim={self.dim_value}, num_heads={self.num_heads}, efficient_attention={self.efficient_attention}"


class SABasicBlock(nn.Module):
    # basic building block of CIFAR SAResNet
    def __init__(self,
                 self_attention,
                 in_planes: int,
                 planes: int,
                 stride: int,
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super().__init__()
        self.sa1 = self_attention(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=1 - 0.9999)
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(stride)
        self.sa2 = self_attention(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=1 - 0.9999)
        self.act = act

        self.downsample = nn.Identity()
        if in_planes != planes:
            self.downsample = nn.Sequential(self_attention(in_planes, planes), nn.MaxPool2d(stride))

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        x = self.sa1(input)
        x = self.pool(x)
        self.bn1(x)
        x = self.act(x)

        x = self.sa2(x)
        self.bn2(x)
        x = self.act(x)

        x += self.downsample(input)
        return self.act(x)


class SAResNet(nn.Module):
    def __init__(self,
                 self_attention,
                 block_size: int,
                 num_classes: int,
                 act: Callable[[torch.Tensor], torch.Tensor] = F.relu
                 ):
        super().__init__()
        self.in_plane = 16
        self.act = act
        self.sa = self_attention(3, self.in_plane)
        self.bn = nn.BatchNorm2d(self.in_plane, momentum=1 - 0.9999)

        self.layer1 = self._make_layer(partial(SABasicBlock, self_attention, act=act), 16, block_size, 1)
        self.layer2 = self._make_layer(partial(SABasicBlock, self_attention, act=act), 32, block_size, 2)
        self.layer3 = self._make_layer(partial(SABasicBlock, self_attention, act=act), 64, block_size, 2)

        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_plane, planes, stride))
            self.in_plane = planes
        return nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = self.sa(x)
        x = self.bn(x)
        x = self.act(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.adaptive_avg_pool2d(x, [1, 1]).flatten(1)
        return self.fc(x)


from homura import Registry

MODEL_REGISTRY = Registry("self-attention models", SAResNet)


@MODEL_REGISTRY.register
def lambda_resnet20(num_classes: int):
    lambda_sa = partial(SelfAttention, lambda_attention=True, dim_key=16, num_heads=4)
    return SAResNet(lambda_sa, 3, num_classes)


@MODEL_REGISTRY.register
def lambda_resnet56(num_classes: int):
    lambda_sa = partial(SelfAttention, lambda_attention=True, dim_key=16, num_heads=4)
    return SAResNet(lambda_sa, 9, num_classes)


@MODEL_REGISTRY.register
def efficient_resnet20(num_classes: int):
    eff_sa = partial(SelfAttention, efficient_attention=True, dim_key=16, num_heads=4)
    return SAResNet(eff_sa, 3, num_classes)


@MODEL_REGISTRY.register
def efficient_resnet56(num_classes: int):
    eff_sa = partial(SelfAttention, efficient_attention=True, dim_key=16, num_heads=4)
    return SAResNet(eff_sa, 9, num_classes)


@MODEL_REGISTRY.register
def dotprod_resnet20(num_classes: int):
    eff_sa = partial(SelfAttention, dim_key=16, num_heads=4)
    return SAResNet(eff_sa, 3, num_classes)


@MODEL_REGISTRY.register
def dotprod_resnet56(num_classes: int):
    eff_sa = partial(SelfAttention, dim_key=16, num_heads=4)
    return SAResNet(eff_sa, 9, num_classes)
