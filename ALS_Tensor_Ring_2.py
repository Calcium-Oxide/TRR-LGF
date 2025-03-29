import torch
from torch import nn
import string

## Set random seed and device
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Set complete
## TR-ALS Decomposition :)
# Class start
class ALS_TR(nn.Module):
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
        self.rank = rank
        self.epoch = 0
    # Initialization complete

    # Select the optimization node and pass the corresponding L2 norm
    def ALS_opt(self, epoch):
        index = epoch % self.N  # get the index of node
        regulation = 0
        for j in range(self.N):
            if j == index:
                self.node[j].requires_grad_(True)  # Select the optimization node
                regulation = torch.norm(self.node[j]) ** 2  # get the corresponding L2 norm
            else:
                self.node[j].requires_grad_(True)
                regulation += torch.norm(self.node[j]) ** 2
        # Generate the string used for tensor contraction of node
        dimensions = ''.join([chr(ord('a') + i) for i in range(self.N)])
        str_parts = []
        for i in range(self.N):
            first_char = chr(ord('z') - i)  
            str_parts.append(first_char + dimensions[i] + chr(ord('z') - (i + 1) % self.N))
        str = ','.join(str_parts)
        einsum_str = f"{str}->{dimensions}"
        # Generation complete
        merged_tensor = torch.einsum(einsum_str, *self.node)  # Restore the node to tensor form
        return merged_tensor, regulation
    # Selection complete

    # Set the string used for tensor contraction of W and X
    def generate_einsum_string(self, dim, n):
        letters = string.ascii_lowercase
        letters_shifted = letters[1:dim + 1] + letters[0]
        einsum_str = letters[:dim - n + 1] + ',' + letters_shifted[:dim] + '->' + letters[0] + letters_shifted[dim - n:dim]
        return einsum_str
    # Generation complete


    # Tensor Regression
    def Tensor_Regression(self, x, epoch):
        einsum_str = self.generate_einsum_string(self.N, self.n)
        w, regulation = self.ALS_opt(epoch)
        y_hat = torch.einsum(einsum_str, x, w)
        bias = torch.ones_like(y_hat)
        y_hat = y_hat + bias 
        return y_hat, regulation
    # Regression complete

    def forward(self, x):
        epoch = self.epoch  # Set up an incrementing counter to control the selection of nodes that need optimization
        self.epoch += 1
        y_hat, regulation = self.Tensor_Regression(x, epoch)
        """""
        dimensions = ''.join([chr(ord('a') + i) for i in range(self.N)])
        str_parts = []
        for i in range(self.N):
            first_char = chr(ord('z') - i)
            second_char = chr(ord('z') - (i + 1) % self.N)
            str_parts.append(first_char + dimensions[i] + second_char)
        str = ','.join(str_parts)
        einsum_str = f"{str}->{dimensions}"
        W = torch.einsum(einsum_str, *self.node)
        """""
        return  y_hat,regulation 
# Class end

"""""
input_shape = torch.randn(1,3,32,32).to(device)
output_shape = torch.randn(1,1,1,5).to(device)
model = ALS_TR(input_shape.shape, output_shape.shape, rank=[5,5,5,5,5,5]).to(device)
W, bias, regulation, y_hat = model(input_shape)
print(y_hat.shape)
"""""