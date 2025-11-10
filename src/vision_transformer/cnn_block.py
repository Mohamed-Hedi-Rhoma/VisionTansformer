import torch
import torch.nn as nn
import torch.nn.functional as F 

class Resblock(nn.Module) : 
    def __init__(self, n_chan):
        super(Resblock,self).__init__()
        self.conv = nn.Conv2d(n_chan,n_chan,kernel_size=3,stride=1,padding=1,bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chan)
        torch.nn.init.kaiming_normal_(self.conv.weight,nonlinearity='relu')
        torch.nn.init.constant_(self.batch_norm.weight,0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)
    def forward(self,x): 
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x
    


class cnn_block(nn.Module):
    def __init__(self):
        super(cnn_block,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1)
        self.resblocks_1 = nn.Sequential(*(3*[Resblock(n_chan=32)]))
        self.conv2 = nn.Conv2d(32,128,kernel_size=3,stride=1,padding=1)
        self.resblocks_2 = nn.Sequential(*(3*[Resblock(n_chan=128)]))
        self.conv3 = nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1)
        self.resblocks_3 = nn.Sequential(*(3*[Resblock(n_chan=256)]))
        self.conv4 = nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1)
        self.resblocks_4 = nn.Sequential(*(3*[Resblock(n_chan=512)]))
    def forward(self,x) : 
        batch_size , n_chann , H,W = x.shape
        
        out = torch.relu(self.conv1(x))
        out = self.resblocks_1(out)
        out = F.max_pool2d(out,2)
        
        out = torch.relu(self.conv2(out))
        out = self.resblocks_2(out)
        out = F.max_pool2d(out,2)
        
        out = torch.relu(self.conv3(out))
        out = self.resblocks_3(out)
        out = F.max_pool2d(out,2)
        
        out = torch.relu(self.conv4(out))
        out = self.resblocks_4(out)
        out = F.max_pool2d(out,2)
        
        out = out.view(batch_size,512,-1)
        out = out.permute(0,2,1)
        return out




"""test_input = torch.randn(3, 3, 224, 224)  # (batch=3, channels=3, H=224, W=224)

# Create your model
model = cnn_block()

# Forward pass
output = model(test_input)

# Check output shape
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")"""