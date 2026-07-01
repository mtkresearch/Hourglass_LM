import torch
import numpy as np
import torch.nn as nn

class SkipAdd(nn.Module):
    '''
    The skip connection module
    '''
    def forward(self, x, y):
        return x + y
    
class MLPBlockReLU(nn.Module):
    '''
    Single MLP block, where both the middle and the final activation functions are ReLU
    h_{i+1} = h_{i} + W2 * ReLU(W1 * Batch_Norm(h_{i})))
    '''
    def __init__(self, d1, d2, block_id=None):
        super().__init__()
        self.block_id = block_id
        self.norm = nn.BatchNorm1d(d1, affine=False)  # Norm without lernable weight/bias        
        self.fc1 = nn.Linear(d1, d2, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d2, d1, bias=False)
        self.skip_add = SkipAdd()
        self.skip_add.block_id = block_id

    def forward(self, x):
        nx = self.norm(x)
        res = self.fc2(self.act(self.fc1(nx)))
        return self.skip_add(x, res)  # skip_add(x, res)


class ConventionalMLP(nn.Module):
    '''
    The whole conventional MLP
    '''
    def __init__(self,
                 input_dim,
                 output_dim, 
                 latent_dim, 
                 hidden_dims, 
                 wo_Win=False, 
                 wo_Wout=False, 
                 fix_Win=False,
                 I_Win=False):
        super().__init__()

        # assert (latent_dim == input_dim) or (latent_dim == output_dim), \
        #     "WARN: The latent dimension should equal input (or output) dimension in conventional MLP"
        assert all(hidden_dim > latent_dim for hidden_dim in hidden_dims), \
            "ERROR: all hidden dimension should be larger than latent dimension in conventional MLP"
                
        # W_in
        self.wo_Win = wo_Win
        self.wo_Wout = wo_Wout
        self.I_Win = I_Win
        self.fix_Win = fix_Win
        
        if not wo_Win:
            self.in_fc = nn.Linear(input_dim, latent_dim, bias=False)
            if self.I_Win:
                print('Initialize Win to Identity matrix')
                nn.init.eye_(self.in_fc.weight)
            if fix_Win:
                print('Set W_in to non-trianable')
                for param in self.in_fc.parameters():
                    param.requires_grad = False
        else:
            self.in_fc = None  # 明確設為 None（可選）

        # MLP blocks
        blocks = []
        for i, hidden_dim in enumerate(hidden_dims):  
            blocks.append(MLPBlockReLU(latent_dim, hidden_dim, block_id=i+1))
        self.blocks = nn.ModuleList(blocks)

        # W_out
        if not wo_Win:
            self.out_fc = nn.Linear(latent_dim, output_dim, bias=False)
        else:
            self.out_fc = None

    def forward(self, x):
        # W_in
        if self.wo_Win:
            h = x
        else:
            h = self.in_fc(x)

        # MLP blocks
        for block in self.blocks:
            h = block(h)
        
        # W_out
        if self.wo_Wout:
            y_pred = torch.sigmoid(h)
        else:   
            y_pred = torch.sigmoid(self.out_fc(h))

        return y_pred
    
class HourGlassMLP(nn.Module):
    '''
    The whole hourglass MLP
    '''
    def __init__(self,
                 input_dim, 
                 output_dim, 
                 latent_dim, 
                 hidden_dims,
                 fix_Win=False):
        super().__init__()

        # assert latent_dim > input_dim, \
        #     "ERROR: latent dimension should be wider than input dimension in HourGlass"
        assert all(hidden_dim < latent_dim for hidden_dim in hidden_dims), \
            "ERROR: all hidden dimensions should be narrower than latent dimension"
        
        # W_in
        self.in_fc = nn.Linear(input_dim, latent_dim, bias=False)
        if fix_Win:
            print('Set W_in to non-trianable')
            for param in self.in_fc.parameters():
                param.requires_grad = False
        
        # MLP blocks
        self.blocks = nn.ModuleList([
            MLPBlockReLU(latent_dim, hidden_dim, block_id=i+1) 
            for i, hidden_dim in enumerate(hidden_dims)
        ])
        
        # W_out
        self.out_fc = nn.Linear(latent_dim, output_dim, bias=False)
        
    def forward(self, x):
        h = self.in_fc(x)
        for block in self.blocks:
            h = block(h)
        y_pred = torch.sigmoid(self.out_fc(h))
        return y_pred