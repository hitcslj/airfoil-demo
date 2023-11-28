import torch
from torch import nn
import math

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
    
    
class AE_B_Attention(nn.Module):
    def __init__(self,in_channels=2,
                 ae_channels=30*16,
                ) -> None:
        super().__init__()

        self.mlp11 = MLP(in_channels,8,16) # (B,20,2) --> (B,20,16)
        self.qkv = nn.Linear(16, 16*3)
        self.mlp12 = MLP(ae_channels,ae_channels//2,ae_channels)
        self.ae = AE(in_channels=ae_channels)
        self.mlp21 = MLP(ae_channels, ae_channels*2, 200*16)
        self.mlp22 = MLP(16, 8, in_channels)
    
    def attention(self,query, key, value, mask=None, dropout=None):
        """
        Scaled Dot Product Attention
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self,x,p,mid_input=None,mid_data=None): # p就是物理参数
        bs = x.shape[0]
        x = torch.cat((x,p),dim=1)
        ae_input = self.mlp11(x) # (B,30,2) --> (B,30,16)
        ## 对ae_input进行attention
        q,k,v = torch.split(self.qkv(ae_input), 16, dim=2)
        
        ae_input, _ = self.attention(q, k, v)
        ae_input = ae_input.reshape(bs,-1) # (B,30,16) --> (B,30*16)
        ae_input = self.mlp12(ae_input) # (B,30*16) --> (B,30*16)
        ae_output = self.ae(ae_input) # (B,30*16) --> (B,30*16)
        ae_output = self.mlp21(ae_output) # (B,30*16) --> (B,200*16)
        y = ae_output.reshape(bs,200,16) # (B,200*16) --> (B,200,16)

        output = self.mlp22(y) # (B,200,16) --> (B,200,2)
        return output

class AE_B_Attention2(nn.Module):
    def __init__(self,in_channels=2,
                 ae_channels=20*16,
                ) -> None:
        super().__init__()

        self.mlp11 = MLP(in_channels,8,16) # (B,20,2) --> (B,20,16)
        self.mlp12 = MLP(in_channels,8,16) # (B,10,2) --> (B,10,16)
        self.q = nn.Linear(16, 16)
        self.kv = nn.Linear(16, 16*2)
        self.mlp13 = MLP(ae_channels,ae_channels//2,ae_channels)
        self.ae = AE(in_channels=ae_channels)
        self.mlp21 = MLP(ae_channels, ae_channels*2, 200*16)
        self.mlp22 = MLP(16, 8, in_channels)
    
    def attention(self,query, key, value, mask=None, dropout=None):
        """
        Scaled Dot Product Attention
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self,x,p,mid_input=None,mid_data=None): # p就是物理参数
        bs = x.shape[0]
        x = self.mlp11(x) # (B,20,2) --> (B,20,16)
        p = self.mlp12(p) # (B,10,2) --> (B,10,16)
        ## 对ae_input进行attention
        k,v = torch.split(self.kv(p), 16, dim=2)
        q = self.q(x)
        
        ae_input, _ = self.attention(q, k, v)
        ae_input = ae_input.reshape(bs,-1) # (B,20,16) --> (B,20*16)
        ae_input = self.mlp13(ae_input) # (B,20*16) --> (B,20*16)
        ae_output = self.ae(ae_input) # (B,20*16) --> (B,20*16)
        ae_output = self.mlp21(ae_output) # (B,20*16) --> (B,200*16)
        y = ae_output.reshape(bs,200,16) # (B,200*16) --> (B,200,16)

        output = self.mlp22(y) # (B,200,16) --> (B,200,2)
        return output


if __name__=='__main__':
    model = AE_B_Attention()
    x = torch.randn(1,20,2)
    p = torch.randn(1,10,2)
    y = model(x,p)
    print(y.shape) # [1,200,2]
    print(model)