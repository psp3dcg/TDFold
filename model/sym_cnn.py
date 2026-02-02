import torch
import torch.nn as nn
import torch.nn.functional as F

class Sym_CNN2d(nn.Module):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: defined graph
    '''
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3):
        super(Sym_CNN2d, self).__init__()

        kernel_size = 3         

        ini_kernel = nn.init.kaiming_normal_(torch.empty([out_channels, in_channels, kernel_size, kernel_size]),mode='fan_out',nonlinearity='relu') 
        ini_bias = nn.init.constant_(torch.empty([1,out_channels,1,1]), val = 0.01) # (B, L, L, d)        
        self.ini_kernel = nn.Parameter(ini_kernel.cuda()) 
        self.ini_bias = nn.Parameter(ini_bias.cuda()) #(B, L, L, d)  

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

    def forward(self, pair):
        # if(return_att):
        #     pair = pair.permute(0,3,1,2)
 #       mani_kernel = torch.bmm(self.ini_kernel, self.ini_kernel.permute(0,2,1)).view(self.out_channels,self.in_channels, self.kernel_size, self.kernel_size)

        # kernel + kernel.T
        mani_kernel = (self.ini_kernel.permute(0,1,3,2)+self.ini_kernel)/2
        conv_res = F.conv2d(pair, mani_kernel, padding = int((self.kernel_size-1)/2)) + self.ini_bias
 ##       conv_res = torch.div(conv_res, torch.norm(conv_res,p=2,dim=[], keepdim=True))
        conv_res = torch.relu(conv_res)
        # if(return_att):
        #     conv_res = conv_res.permute(0,2,3,1)
        return conv_res


class Sym_CNN2d_Block(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size= 3, p_drop=0.5):
        super(Sym_CNN2d_Block, self).__init__()
        layer_s = list()

        layer_s.append(Sym_CNN2d(in_channels, hidden_channels, kernel_size))
        layer_s.append(nn.InstanceNorm2d(hidden_channels, affine=True, eps=1e-6))
        layer_s.append(nn.ELU())
        # layer_s.append(nn.Dropout(p_drop))
        layer_s.append(Sym_CNN2d(hidden_channels, hidden_channels, kernel_size))
        layer_s.append(nn.InstanceNorm2d(hidden_channels, affine=True, eps=1e-6))

        self.layer = nn.Sequential(*layer_s)
        self.final_activation = nn.ELU()

    def forward(self, x):
        out = self.layer(x)
        return self.final_activation(x + out)

class Sym_CNN2d_Network(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, n_blocks=2, kernel_size= 3, p_drop=0.5):
        super(Sym_CNN2d_Network, self).__init__()
        layer_s = list()

        for i_block in range(n_blocks):
            layer_s.append(Sym_CNN2d_Block(in_channels, hidden_channels, kernel_size, p_drop))

        layer_s.append(Sym_CNN2d(hidden_channels, out_channels, kernel_size))
        self.layer = nn.Sequential(*layer_s)

    def forward(self, x):
        output = self.layer(x)
        return output