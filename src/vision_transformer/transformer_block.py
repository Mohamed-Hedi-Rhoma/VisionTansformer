import torch
import torch.nn as nn
import torch.nn.functional as F 


class TransformerBlock(nn.Module) : 
    def __init__(self,n_embd , num_heads = 6 , n_hidden = 64):
        super().__init__()
        assert n_embd%num_heads == 0 , "Embedding dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = n_embd// num_heads
        
        
        self.query  = nn.Linear(n_embd , n_embd)
        self.value = nn.Linear(n_embd,n_embd)
        self.key = nn.Linear(n_embd,n_embd)

        self.mlp = nn.Sequential(

            nn.Linear(n_embd,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,n_embd)
        )
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
    def forward(self,x) : 
        batch_size , n_patches , n_embd = x.shape 
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        q = q.view(batch_size , n_patches , self.num_heads , self.head_dim).transpose(1,2)
        k = k.view(batch_size , n_patches , self.num_heads , self.head_dim).transpose(1,2)
        v = v.view(batch_size , n_patches , self.num_heads , self.head_dim).transpose(1,2)

        atten_weights = F.scaled_dot_product_attention(q,k,v)
        atten_weights = atten_weights.transpose(1,2).contiguous().view(batch_size,n_patches , -1)

        out = self.norm1(x+atten_weights)
        x = self.mlp(out) + out
        x = self.norm2(x)

        return x
    


"""test_input = torch.randn( 3, 196, 512)  

# Create your model
model = TransformerBlock(n_embd=512,n_hidden=32 ,num_heads=4)

# Forward pass
output = model(test_input)

# Check output shape
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")"""