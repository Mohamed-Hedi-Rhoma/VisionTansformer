import torch
import torch.nn as nn
import torch.nn.functional as F 


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads=6, mlp_ratio=4, dropout=0.1):
        super().__init__()
        assert n_embd % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads
        
        # Multi-head attention
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd) 
        
        # MLP
        n_hidden = int(n_embd * mlp_ratio)  
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.GELU(),  
            nn.Dropout(dropout),  
            nn.Linear(n_hidden, n_embd)
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        
        # Dropout
        self.dropout_attn = nn.Dropout(dropout)  
        self.dropout_mlp = nn.Dropout(dropout)  
    def forward(self, x):
        batch_size, n_patches, n_embd = x.shape
        
        # Multi-head attention with residual
        norm_x = self.norm1(x)
        q = self.query(norm_x)
        k = self.key(norm_x)
        v = self.value(norm_x)
        
        q = q.view(batch_size, n_patches, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_patches, self.num_heads, self.head_dim).transpose(1, 2)
        
        atten_output = F.scaled_dot_product_attention(q, k, v)
        atten_output = atten_output.transpose(1, 2).contiguous().view(batch_size, n_patches, -1)
        atten_output = self.proj(atten_output)  # ← Project
        atten_output = self.dropout_attn(atten_output)  # ← Dropout
        x = x + atten_output  # Residual
        
        # MLP with residual
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        mlp_out = self.dropout_mlp(mlp_out)  # ← Dropout
        x = x + mlp_out  # Residual
        
        return x
    


"""test_input = torch.randn( 3, 196, 512)  

# Create your model
model = TransformerBlock(n_embd=512,n_hidden=32 ,num_heads=4)

# Forward pass
output = model(test_input)

# Check output shape
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")"""