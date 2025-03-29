import torch
import torch.nn as nn
import torch.nn.functional as F

class SEKG(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.conv_sa = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2) 

    def forward(self, input_x):
        sa_x = self.conv_sa(input_x)  
        y = self.avg_pool(input_x)
        ca_x = self.conv_ca(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).contiguous()
        out  = sa_x + ca_x
        return out
class AFG(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(AFG, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.sekg = SEKG(in_channels, kernel_size)
        self.conv = nn.Conv2d(in_channels, in_channels*kernel_size*kernel_size, 1, 1, 0)

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        x = self.sekg(input_x)
        x = self.conv(x)
        filter_x = x.reshape([b, c, self.kernel_size*self.kernel_size, h, w])

        return filter_x

class DyConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(DyConv, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.afg = AFG(in_channels, kernel_size)
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        
    def forward(self, input_x):
        b, c, h, w = input_x.size()
        filter_x = self.afg(input_x)
        unfold_x = self.unfold(input_x).reshape(b, c, -1, h, w)
        out = (unfold_x * filter_x).sum(2)
        return out


class Dy_scale_Conv(nn.Module):
    def __init__(self,num_levels,inchannels,outchannels):
        super(Dy_scale_Conv,self).__init__()
        self.num_levels = num_levels
        self.in_channels = inchannels
        self.dyconv = DyConv(inchannels)
       # self.conv_out = nn.Conv2d(int(num_levels * inchannels), outchannels,1)
        self.conv_out = nn.Conv2d(int(num_levels*inchannels),outchannels,3,1,1)
        
    def forward(self,input_tensor):
        tensor = input_tensor
        pyramid = [input_tensor]
        for i in range(self.num_levels - 1):
            out = F.interpolate(pyramid[-1], scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            blurred = out
            out = self.dyconv(out)
            out = F.interpolate(out, scale_factor=2*(i+1), mode='bilinear', align_corners=False)
            tensor = torch.cat((tensor, out), dim=1)
            pyramid.append(blurred)
        
        tensor = self.conv_out(tensor)    
        return tensor
    
class Scale_Conv(nn.Module):
    def __init__(self,num_levels,inchannels,outchannels):
        super(Scale_Conv,self).__init__()
        self.num_levels = num_levels
        self.in_channels = inchannels
        self.conv = nn.Conv2d(inchannels,inchannels,3,1,1)
        self.conv_out = nn.Conv2d(int(num_levels*inchannels),outchannels,3,1,1)
        
    def forward(self,input_tensor):
        tensor = input_tensor
        pyramid = [input_tensor]
        for i in range(self.num_levels - 1):
            out = F.interpolate(pyramid[-1], scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            blurred = out
            out = self.conv(out)
            out = F.interpolate(out, scale_factor=2*(i+1), mode='bilinear', align_corners=False)
            tensor = torch.cat((tensor, out), dim=1)
            pyramid.append(blurred)
        
        tensor = self.conv_out(tensor)    
        return tensor
    
class Dy_Conv(nn.Module):
    def __init__(self,num_levels,inchannels,outchannels):
        super(Dy_Conv,self).__init__()
        self.num_levels = num_levels
        self.in_channels = inchannels
        self.dyconv = DyConv(inchannels)
        self.conv_out = nn.Conv2d(inchannels,outchannels,3,1,1)
        
    def forward(self,input_tensor):
        tensor = self.dyconv(input_tensor)
        tensor = self.conv_out(tensor)    
        return tensor

"""""
input = torch.randn(1,3,32,32)
model = scale_feature(num_levels=3,in_channels=3)
output = model(input)
print(output.shape)
# 计算可学习参数量的函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# 计算并打印可学习参数量
num_params = count_parameters(model)
print(f"可学习参数量: {num_params}")
"""""