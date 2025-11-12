import torch
import torch.nn as nn
import torch.nn.functional as F 
import sys
#sys.path.append('/home/mrhouma/Documents/VisionTransformer/Vision_Transformer/src')
from vision_transformer.cnn_block import cnn_block 
from vision_transformer.transformer_block import TransformerBlock


class visiontransformer(nn.Module):
    def __init__(self, n_embd, n_classes, block_size=257, num_blocks=6, 
                 num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        
        # Tokens and embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))  
        self.postional_embd = nn.Embedding(block_size, n_embd)
        self.register_buffer('pos_indices', torch.arange(block_size))  
        
        # Backbone
        self.cnn_block = cnn_block()
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(n_embd=n_embd, num_heads=num_heads, 
                              mlp_ratio=mlp_ratio, dropout=dropout)  
              for _ in range(num_blocks)]
        )
        
        # Classification head
        self.norm_final = nn.LayerNorm(n_embd)  # ← Added
        self.dropout_final = nn.Dropout(dropout)  # ← Added
        self.output_classes = nn.Linear(n_embd, n_classes)
    def forward(self, x):
        batch_size = x.shape[0]
        
        # CNN feature extraction
        x = self.cnn_block(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        seq_len = x.shape[1]
        x = x + self.postional_embd(self.pos_indices[:seq_len])  
        
        # Transformer blocks
        x = self.transformer_blocks(x)
        
        # Classification head
        x = self.norm_final(x)  
        cls_output = x[:, 0, :]
        cls_output = self.dropout_final(cls_output)  
        output = self.output_classes(cls_output)
        
        return output
    


"""test_input = torch.randn(3, 3, 224, 224)  # (batch=3, channels=3, H=224, W=224)

# Create your model
model = visiontransformer(n_embd=512,n_classes=3)

# Forward pass
output = model(test_input)

# Check output shape
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")"""