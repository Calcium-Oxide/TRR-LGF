import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
"""""
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.size()


        # 提取查询、键和值特征图
        query = self.query_conv(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, height * width)
        value = self.value_conv(x).view(batch, -1, height * width)

        # 计算注意力图
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        # 加权值特征图
        out = torch.bmm(value, attention.permute(0, 2, 1))

        # 重塑输出特征图的形状
        out = out.view(batch, channels, height, width)
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )


    def forward(self, x):
        batch, channels, _, _ = x.size()

        # 全局平均池化
        y = self.global_avgpool(x).view(batch, channels)
        # print(y.shape)

        # 通过全连接层获取通道注意力权重
        attention = self.fc(y).view(batch, channels, 1, 1)
        # print(attention.shape)

        # 加权特征图
        out = x * attention
        # print(out.shape)

        return out
class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        #self.spatial_attention = SpatialAttention(in_channels)
        self.channel_attention = ChannelAttention(in_channels)
        self.conv_out = nn.Conv2d(in_channels, in_channels, 1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # 计算空间注意力特征图
        #spatial_out = self.spatial_attention(x)
        # 计算通道注意力特征图
        channel_out = self.channel_attention(x)
        # 将两个特征图相加
        #fused_out= spatial_out + channel_out
        fused_out = self.conv_out(channel_out)
        # 卷积后使用ReLU激活函数
        fused_out = self.relu(fused_out)
        return fused_out
"""""
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert (
            self.head_dim * num_heads == in_channels
        ), "in_channels must be divisible by num_heads"
        
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.fc_out = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.view(N, C, -1).permute(0, 2, 1)  # (N, H*W, C)

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        queries = queries.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (N, num_heads, H*W, head_dim)
        keys = keys.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])  # (N, num_heads, H*W, H*W)
        attention = F.softmax(energy / (self.head_dim ** 0.5), dim=3)

        out = torch.einsum("nhql,nhld->nhqd", [attention, values]).reshape(N, -1, C)
        out = self.fc_out(out)

        return out.view(N, C, H, W)
    
class SelfAttention2(nn.Module):
    def __init__(self, in_channels,num_layers,num_heads):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(in_channels,num_heads)
        self.block = nn.ModuleList([MultiHeadSelfAttention(in_channels, num_heads) for _ in range(num_layers)])
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        for block in self.block:
            x = block(x)
        x = self.drop(x)
        return x   
"""""
class SelfAttention_patch(nn.Module):
    def __init__(self, in_channels, num_heads=4, patch_size=16):
        super(SelfAttention_patch, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.patch_size = patch_size
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.fc_out = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        N, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image size must be divisible by the patch size"
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().reshape(N, C, -1, self.patch_size * self.patch_size).permute(0, 2, 1, 3)
        patches = patches.reshape(N, -1, C)

        queries = self.query(patches)
        keys = self.key(patches)
        values = self.value(patches)

        queries = queries.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])
        attention = F.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nhld->nhqd", [attention, values]).reshape(N, -1, C)
        out = self.fc_out(out)

        out = out.view(N, C, H // self.patch_size, W // self.patch_size, self.patch_size, self.patch_size)
        out = out.permute(0, 1, 2, 4, 3, 5).contiguous().view(N, C, H, W)

        return out
    
"""""