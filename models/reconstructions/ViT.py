import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % self.num_heads == 0, "embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        b, h, w, _ = x.size()
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = rearrange(q, 'b h w (n d) -> b n h w d', n=self.num_heads)
        k = rearrange(k, 'b h w (n d) -> b n h w d', n=self.num_heads)
        v = rearrange(v, 'b h w (n d) -> b n h w d', n=self.num_heads)

        k_T = k.transpose(-2, -1)
        scores = torch.matmul(q, k_T) / self.head_dim**0.5
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)

        out = rearrange(out, 'b n h w d -> b h w (n d)')
        out = self.out(out)
        return out

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = rearrange(x, 'b h w d -> b (h w) d')
        x = self.mlp(x)
        x = rearrange(x, 'b (h w) d -> b h w d', h=int(x.size(1) ** 0.5))
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, num_heads, mlp_dim):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attention = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x):
        out = self.self_attention(self.norm1(x))
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, inplanes=3, num_classes=12, hidden_dim=256, num_heads=4, mlp_dim=128):
        super(ViT, self).__init__()
        self.patch_size = 1
        self.num_patches = inplanes * 14 * 14
        self.embedding_dim = hidden_dim

        self.conv1 = nn.Conv2d(inplanes, hidden_dim, kernel_size=self.patch_size, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.vit_blocks = nn.ModuleList([
            ViTBlock(hidden_dim, self.num_patches, hidden_dim, num_heads, mlp_dim)
            for _ in range(4)
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        
        out = self.conv1(x["image"])
        out = self.bn1(out)
        out = self.relu(out)

        outputs_map = [] 
        
        out = rearrange(out, 'b c h w -> b h w c')
        for vit_block in self.vit_blocks:
            out = vit_block(out)
            outputs_map.append(out)
        
        out = rearrange(out, 'b h w c -> b c h w')

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        outputs_map = [x.detach() for x in outputs_map]
       
        return {"class_out": out, "outputs_map": outputs_map}


# # Instantiate the ViT network with three layers
# vit_net = ViT(inplanes=64, instrides=None, num_classes=15)
# print(vit_net)