import torch
import torch.nn as nn
import string
import torch.nn.functional as F


## Set random seed and device
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Set complete

class FPN(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(FPN, self).__init__()
        self.in_channels = inchannels
        self.conv = nn.Conv2d(inchannels,inchannels,3,1,1)
        self.conv_out = nn.Conv2d(int(3*inchannels),outchannels,3,1,1)
        
    def forward(self,input_tensor):
        tensor = input_tensor
        pyramid = [input_tensor]
        for i in range(2):
            out = F.interpolate(pyramid[-1], scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            blurred = out
            out = self.conv(out)
            out = F.interpolate(out, scale_factor=2*(i+1), mode='bilinear', align_corners=False)
            tensor = torch.cat((tensor, out), dim=1)
            pyramid.append(blurred)
        
        out = self.conv_out(tensor)    
        return out

class ConvLayer(nn.Module):
    def __init__(self,outchannel):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,256,3,1,1)
        self.bn2 = nn.BatchNorm2d(256)
        self.fpn = FPN(inchannels=256,outchannels=outchannel)
        self.bn3 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fpn(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
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

class SelfAttention(nn.Module):
    def __init__(self, in_channels,num_layers,num_heads):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(in_channels,num_heads)
        self.block = nn.ModuleList([MultiHeadSelfAttention(in_channels, num_heads) for _ in range(num_layers)])
        self.drop = nn.Dropout2d(0.15)

    def forward(self, x):
        for block in self.block:
            x = block(x)
        x = self.drop(x)
        return x 
## TR-ALS Decomposition :)
# Class start
class TR(nn.Module):
    # Initialize the ALS_TR class
    def __init__(self, input_shape, output_shape, rank):
        super().__init__()
        shape_w = input_shape[1:] + output_shape[1:]  # Obtain the shape of W
        self.N = len(shape_w) 
        self.n = len(output_shape) - 1
        self.node = []  
        
        for n in range(self.N):
            # Get the current rank values
            current_rank = rank[n % len(rank)]  # Use modulo to cycle through the rank list
            next_rank = rank[(n + 1) % len(rank)]  # Get the next rank, cycling back to the start if needed
            shape = (current_rank, shape_w[n], next_rank)
            parameter = nn.Parameter(torch.randn(shape, requires_grad=True).to(device))
            self.node.append(parameter)  # Assign parameters to each node
            self.register_parameter(f'node_{n}', parameter)  # Register parameters to ensure they are recognized and updated
        # Prepare for generating parameters for tensor ring end
    # Initialization complete

    # Select the optimization node and pass the corresponding L2 norm
    def Merge(self):
        dimensions = ''.join([chr(ord('a') + i) for i in range(self.N)])
        str_parts = []
        for i in range(self.N):
            first_char = chr(ord('z') - i)  
            str_parts.append(first_char + dimensions[i] + chr(ord('z') - (i + 1) % self.N))
        str = ','.join(str_parts)
        einsum_str = f"{str}->{dimensions}"
        # Generation complete
        merged_tensor = torch.einsum(einsum_str, *self.node)  # Restore the node to tensor form
        return merged_tensor
    # Selection complete

    # Set the string used for tensor contraction of W and X
    def generate_einsum_string(self, dim, n):
        letters = string.ascii_lowercase
        letters_shifted = letters[1:dim + 1] + letters[0]
        einsum_str = letters[:dim - n + 1] + ',' + letters_shifted[:dim] + '->' + letters[0] + letters_shifted[dim - n:dim]
        return einsum_str
    # Generation complete


    # Tensor Regression
    def Tensor_Regression(self, x):
        einsum_str = self.generate_einsum_string(self.N, self.n)
        w = self.Merge()
        y_hat = torch.einsum(einsum_str, x, w)
        bias = torch.ones_like(y_hat)
        y_hat = y_hat + bias 
        return y_hat
    # Regression complete

    def forward(self, x):
        y_hat = self.Tensor_Regression(x)
        return y_hat
# Class end
class CATR(nn.Module):
    def __init__(self,in_shape, out_shape, rank, num_layers, m_channel, num_head=8):
        super(CATR, self).__init__()
        self.conv_layer = ConvLayer(outchannel=m_channel)
        self.self_attention = SelfAttention(m_channel,num_layers, num_head)
        self.als_tr = TR(in_shape, out_shape, rank)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.self_attention(x)
        y_hat = self.als_tr(x)
        return y_hat
    
#input = torch.randn(1,3,32,32).to(device)
#model = CATR((1,512,8,8), (1,1,1,5), rank=[5,5,5,5,5,5],num_layers=1, m_channel=512, num_head=8).to(device)
#y_hat = model(input)
#print(y_hat.shape)