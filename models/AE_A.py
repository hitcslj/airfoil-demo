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


class AE_A_Parsec(nn.Module): # source_param,target_param,source_keypoint -> target_keypoint
    def __init__(self,in_channels=2,
                 ae_channels=26*16,
                ) -> None:
        super().__init__()

        self.mlp11 = MLP(in_channels,4,2)  
        self.mlp12 = MLP(2,4,16)
        
        self.mlp_delta1 = MLP(1,4,16)
        self.mlp_delta2 = MLP(11*16,15*16,26*16)
        
        self.input_proj = MLP(26*16,26*16,26*16)
        
        self.ae = AE(in_channels=ae_channels)
        self.mlp21 = MLP(ae_channels, ae_channels*2, 26*16)
        self.mlp22 = MLP(16, 8, 2)
        
    def _forward_with_cond(self,x,cond,bs):
        x = x.reshape(bs,-1) # (B,26,16) --> (B,26*16)
        x = self.input_proj(x) # (B,26*16) --> (B,26*16)
        return x + cond 
        
    def forward(self,p,p2,x): 
        bs = p.shape[0]
        delta_p = torch.sub(p,p2) # (B,11,1)
        delta_feat = self.mlp_delta1(delta_p) # (B,11,16)
        delta_feat = delta_feat.reshape(bs,-1) # (B,11*16)
        delta_feat = self.mlp_delta2(delta_feat) # (B,11*16) --> (B,26*16)
        
        ae_input = self.mlp11(x) # (B,26,2) --> (B,26,2)
        ae_input = self.mlp12(ae_input) # (B,26,2) --> (B,26,16)
        ae_input = self._forward_with_cond(ae_input,delta_feat,bs)
        
        ae_output = self.ae(ae_input) # (B,26*16) --> (B,26*16)
        ae_output = self.mlp21(ae_output) # (B,26*16) --> (B,26*16)
        y = ae_output.reshape(bs,26,16) # (B,26*16) --> (B,26,16)

        output = self.mlp22(y) # (B,26,16) --> (B,26,2)
        return output


class AE_A_Keypoint(nn.Module): # （source_keypoint, source_param, target_keypoint）-> target_param
    def __init__(self,in_channels=2,
                 ae_channels=11*16,
                ) -> None:
        super().__init__()

        self.mlp11 = MLP(in_channels,4,2) # (B,257,2) --> (B,257,16)
        self.mlp12 = MLP(2,4,16)
        
        self.mlp_delta1 = MLP(2,4,16)
        self.mlp_delta2 = MLP(26*16,15*16,11*16)
        
        self.input_proj = MLP(11*16,11*16,11*16)
        
        self.ae = AE(in_channels=ae_channels)
        self.mlp21 = MLP(ae_channels, ae_channels*2, 11*16)
        self.mlp22 = MLP(16, 8, 1)
        
    def _forward_with_cond(self,x,cond,bs):
        x = x.reshape(bs,-1) # (B,11,16) --> (B,11*16)
        x = self.input_proj(x) # (B,11*16) --> (B,11*16)
        return x + cond 
        
    def forward(self,x,x2,p): 
        bs = x.shape[0]
        delta_x = torch.sub(x,x2) # (B,26,2)
        delta_feat = self.mlp_delta1(delta_x) # (B,26,16)
        delta_feat = delta_feat.reshape(bs,-1) # (B,26*16)
        delta_feat = self.mlp_delta2(delta_feat) # (B,26*16) --> (B,11*16)
        
        ae_input = self.mlp11(p) # (B,11,2)
        ae_input = self.mlp12(ae_input) # (B,11,2) --> (B,11,16)
        ae_input = self._forward_with_cond(ae_input,delta_feat,bs)
        
        ae_output = self.ae(ae_input) # (B,11*16) --> (B,11*16)
        ae_output = self.mlp21(ae_output) # (B,11*16) --> (B,11*16)
        y = ae_output.reshape(bs,11,16) # (B,11*16) --> (B,11,16)

        output = self.mlp22(y) # (B,11,16) --> (B,10,1)
        return output



