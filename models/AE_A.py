import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim) -> None:
        super(MLP,self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)  # 2个隐层
        self.relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self,x):
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class AE(nn.Module):
    def __init__(self, in_channels=200*16):
        super(AE, self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(in_channels,256),
            nn.LeakyReLU(),
            nn.Linear(256,64),
            nn.LeakyReLU(),
            nn.Linear(64,20),
            nn.LeakyReLU()
        )
        self.decoder=nn.Sequential(
            nn.Linear(20,64),
            nn.LeakyReLU(),
            nn.Linear(64,256),
            nn.LeakyReLU(),
            nn.Linear(256,in_channels),
            nn.Sigmoid()
        )
    def forward(self,x):       
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded
 
class AE_A(nn.Module):
    def __init__(self,in_channels=2,
                 ae_channels=50*16,
                ) -> None:
        super().__init__()

        self.mlp11 = MLP(in_channels,8,16) # (B,200,2) --> (B,200,16)
        self.mlp12 = MLP(ae_channels,ae_channels//2,ae_channels)
        self.ae = AE(in_channels=ae_channels)
        self.mlp21 = MLP(ae_channels, ae_channels*2, 10*16)
        self.mlp22 = MLP(16, 8, 1)

    def forward(self,x,x2,p): # y就是物理参数
        bs = x.shape[0]
        x = torch.cat((x,x2,p),dim=1)
        ae_input = self.mlp11(x) # (B,50,2) --> (B,50,16)
        ae_input = ae_input.reshape(bs,-1) # (B,50,16) --> (B,50*16)
        ae_input = self.mlp12(ae_input) # (B,50*16) --> (B,50*16)
        ae_output = self.ae(ae_input) # (B,50*16) --> (B,50*16)
        ae_output = self.mlp21(ae_output) # (B,50*16) --> (B,10*16)
        y = ae_output.reshape(bs,10,16) # (B,10*16) --> (B,10,16)

        output = self.mlp22(y) # (B,10,16) --> (B,10,1)
        return output

class AE_A_variable(nn.Module):
    def __init__(self,in_channels=1,
                 ae_channels=10*16,
                ) -> None:
        super().__init__()

        self.mlp11 = MLP(in_channels,4,2) # (B,200,2) --> (B,200,16)
        self.mlp12 = MLP(2,4,16)
        
        self.mlp_delta1 = MLP(2,4,16)
        self.mlp_delta2 = MLP(20*16,15*16,10*16)
        
        self.input_proj = MLP(10*16,10*16,10*16)
        
        self.ae = AE(in_channels=ae_channels)
        self.mlp21 = MLP(ae_channels, ae_channels*2, 10*16)
        self.mlp22 = MLP(16, 8, 1)
        
    def _forward_with_cond(self,x,cond,bs):
        x = x.reshape(bs,-1) # (B,10,16) --> (B,10*16)
        x = self.input_proj(x) # (B,10*16) --> (B,10*16)
        return x + cond 
        
    def forward(self,x,x2,p): # y就是物理参数
        bs = x.shape[0]
        delta_x = torch.sub(x,x2) # (B,20,2)
        delta_feat = self.mlp_delta1(delta_x) # (B,20,16)
        delta_feat = delta_feat.reshape(bs,-1) # (B,20*16)
        delta_feat = self.mlp_delta2(delta_feat) # (B,20*16) --> (B,10*16)
        
        ae_input = self.mlp11(p) # (B,10,1) --> (B,10,2)
        ae_input = self.mlp12(ae_input) # (B,10,2) --> (B,10,16)
        ae_input = self._forward_with_cond(ae_input,delta_feat,bs)
        
        ae_output = self.ae(ae_input) # (B,10*16) --> (B,10*16)
        ae_output = self.mlp21(ae_output) # (B,10*16) --> (B,10*16)
        y = ae_output.reshape(bs,10,16) # (B,10*16) --> (B,10,16)

        output = self.mlp22(y) # (B,10,16) --> (B,10,1)
        return output