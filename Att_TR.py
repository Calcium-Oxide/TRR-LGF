import torch
import torch.nn as nn
#from Conv_Layer import ConvLayer2
from Conv_Layer import ConvLayer3
#from Conv_Layer import ConvLayer4
#from Conv_Layer import ConvLayer5
from Attention import SelfAttention2
from ALS_Tensor_Ring_2 import ALS_TR
from TR_NoD import TR_NoD
from MLP import MLPClassifier

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Att_Tr(nn.Module):
    def __init__(self,in_shape, out_shape, rank, num_layers, m_channel, num_head=8):
        super(Att_Tr, self).__init__()
        self.conv_layer = ConvLayer3(outchannel=m_channel)
        self.self_attention = SelfAttention2(m_channel,num_layers, num_head)
        self.als_tr = ALS_TR(in_shape, out_shape, rank)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.self_attention(x)
        y_hat,regulation = self.als_tr(x)
        return y_hat, regulation
"""""
class Att_Tr_NoD(nn.Module):
    def __init__(self, in_shape, out_shape, rank, num_layers, m_channel, num_head=8):
        super(Att_Tr_NoD, self).__init__()
        self.conv_layer = ConvLayer3(outchannel=m_channel)
        self.self_attention = SelfAttention2(m_channel, num_layers, num_head)
        self.tr_nod = TR_NoD(in_shape, out_shape)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.self_attention(x)
        y_hat = self.tr_nod(x)
        return y_hat

class Att_MLP(nn.Module):
    def __init__(self, in_shape, out_shape, rank, num_layers, m_channel, num_head=8):
        super(Att_MLP, self).__init__()
        self.conv_layer = ConvLayer3(outchannel=m_channel)
        self.self_attention = SelfAttention2(m_channel, num_layers, num_head)
        self.mlp = MLPClassifier((64,512,8,8), num_age_classes=1, num_gender_classes=2, num_race_classes=5)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.self_attention(x)
        y_hat = self.mlp(x)
        return y_hat
    
class Att_Tr_NOPDC(nn.Module):
    def __init__(self,in_shape, out_shape, rank , num_layers, m_channel, num_head=8):
        super(Att_Tr_NOPDC, self).__init__()
        self.conv_layer = ConvLayer3(outchannel=m_channel)
        self.self_attention = SelfAttention2(m_channel,num_layers, num_head)
        self.als_tr = ALS_TR(in_shape, out_shape, rank)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.self_attention(x)
        y_hat,regulation = self.als_tr(x)
        return y_hat,regulation
"""""