import torch
from torch import nn
import string

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TR_NoD(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        shape_w = input_shape[1:] + output_shape[1:]
        self.N = len(shape_w)
        self.n = len(output_shape) - 1
        parameter = nn.Parameter(torch.randn(shape_w, requires_grad=True).to(device))
        self.W = parameter
        self.register_parameter('W', parameter)


    def generate_einsum_string(self, dim, n):
        letters = string.ascii_lowercase
        letters_shifted = letters[1:dim + 1] + letters[0]
        einsum_str = letters[:dim - n + 1] + ',' + letters_shifted[:dim] + '->' + letters[0] + letters_shifted[dim - n:dim]
        return einsum_str

    def Tensor_Regression(self, x):
        einsum_str = self.generate_einsum_string(self.N, self.n)
        y_hat = torch.einsum(einsum_str, x, self.W)
        bias = torch.ones_like(y_hat)
        y_hat = y_hat + bias
        return y_hat

    def forward(self, x):
        y_hat = self.Tensor_Regression(x)
        return y_hat

"""""
data = torch.randn(1,3,32,32).to(device)
output = torch.randn(1,10).to(device)
model = TR_NoD(data.shape, output.shape)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params}")
y_hat = model(data).to(device)
print(y_hat.shape)
"""""