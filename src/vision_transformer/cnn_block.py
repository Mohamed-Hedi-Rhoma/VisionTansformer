import torch
import torch.nn as nn
import torch.nn.functional as F 

class Resblock(nn.Module) : 
    def __init__(self, n_chan , dropout = 0.2):
        super(Resblock,self).__init__()
        self.conv = nn.Conv2d(n_chan,n_chan,kernel_size=3,stride=1,padding=1,bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chan)
        self.dropout = nn.Dropout(dropout)
        torch.nn.init.kaiming_normal_(self.conv.weight,nonlinearity='relu')
        torch.nn.init.zeros_(self.batch_norm.bias)
    def forward(self,x): 
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        out = self.dropout(out)  
        return out + x
    


class cnn_block(nn.Module):
    def __init__(self):
        super(cnn_block, self).__init__()
        
        # Stage 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.resblocks_1 = nn.Sequential(*(2*[Resblock(n_chan=32, dropout=0.1)]))
        
        # Stage 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        self.resblocks_2 = nn.Sequential(*(2*[Resblock(n_chan=64, dropout=0.15)]))
        
        # Stage 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        self.resblocks_3 = nn.Sequential(*(3*[Resblock(n_chan=128, dropout=0.2)]))
        
        # Stage 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        torch.nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        self.resblocks_4 = nn.Sequential(*(3*[Resblock(n_chan=256, dropout=0.25)]))
    
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        
        out = self.bn1(self.conv1(x))
        out = torch.relu(out)
        out = self.resblocks_1(out)
        
        out = self.bn2(self.conv2(out))
        out = torch.relu(out)
        out = self.resblocks_2(out)
        
        out = self.bn3(self.conv3(out))
        out = torch.relu(out)
        out = self.resblocks_3(out)
        
        out = self.bn4(self.conv4(out))
        out = torch.relu(out)
        out = self.resblocks_4(out)
        
        out = F.max_pool2d(out, 2)
        out = out.view(batch_size, 256, -1)
        out = out.permute(0, 2, 1)
        
        return out




"""test_input = torch.randn(3, 3, 224, 224)  # (batch=3, channels=3, H=224, W=224)

# Create your model
model = cnn_block()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Forward pass
output = model(test_input)

# Check output shape
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")"""