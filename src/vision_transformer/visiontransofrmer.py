import torch
import torch.nn as nn
import torch.nn.functional as F 
import sys
#sys.path.append('/home/mrhouma/Documents/VisionTransformer/Vision_Transformer/src')
from vision_transformer.cnn_block import cnn_block 
from vision_transformer.transformer_block import TransformerBlock


class visiontransformer(nn.Module) : 
    def __init__(self , n_embd , n_classes , block_size = 197 , num_blocks = 2 , n_hidden =32 ,num_heads = 2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd))
        self.postional_embd = nn.Embedding(block_size,n_embd)
        self.cnn_block = cnn_block()
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_hidden, num_heads) for _ in range(num_blocks)]
        )
        self.output_classes = nn.Linear(n_embd, n_classes)
    def forward(self , x ) : 
        batch_size , n_channels , H ,W = x.shape 
        x = self.cnn_block(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  
        seq_len = x.shape[1]
        x = x + self.postional_embd(torch.arange(seq_len))
        x = self.transformer_blocks(x)
        cls_output = x[:, 0, :]
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