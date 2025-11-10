import torch 
from vision_transformer.dataset.dataset import Plant_disease_dataset 
import sys
sys.path.append('/home/mrhouma/Documents/VisionTransformer/Vision_Transformer/src')
from vision_transformer.visiontransofrmer import visiontransformer
import torch.optim as optim
from torch.utils.data import Dataset , DataLoader , random_split
import torch.nn.functional as F 
import datetime
from tqdm import tqdm
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
data =Plant_disease_dataset(path_data="/home/mrhouma/Documents/VisionTransformer/Vision_Transformer/ai_training_data")
total = len(data)
train_size = int(0.8 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size
train_dataset, valid_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])
train_data_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)
valid_data_loader = DataLoader(valid_dataset,batch_size=8,shuffle=False)


n_epochs = 5 
lr_rate = 0.01 
patience = 3
best_val_loss = float('inf') 
patience_counter = 0  


model = visiontransformer(n_embd=512,n_classes=2)
optimizer = optim.Adam(params=model.parameters(),lr=lr_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # Minimize validation loss
    patience=2,      # Wait 2 epochs before reducing
    factor=0.5,      # Multiply LR by 0.5
)

for epoch in range (1,n_epochs+1) : 

    start_time = datetime.datetime.now()
    loss_train = 0.0
    correct_train = 0
    total_train = 0

    # Wrap dataloader with tqdm
    pbar = tqdm(train_data_loader, desc=f'Epoch {epoch}/{n_epochs}')
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        x , labels = batch
        logits = model(x)
        loss = F.cross_entropy(logits,labels)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        _, predicted = torch.max(logits, dim=1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        
        # Update progress bar with current metrics
        pbar.set_postfix({
            'loss': loss.item(),  # Current batch loss
            'avg_loss': loss_train / (pbar.n + 1),  # Running average
            'acc': 100 * correct_train / total_train  # Running accuracy
        })

    end_time = datetime.datetime.now()
    epoch_duration = (end_time - start_time).total_seconds()
    if epoch == 1 or epoch % 2 == 0:
        print('{} Epoch {}, Training loss {:.6f}, Time {:.2f}s'.format(
        datetime.datetime.now(), epoch,
        loss_train / len(train_data_loader), epoch_duration))
   
    with torch.no_grad():
            val_loss = 0.0 
            correct = 0
            total = 0
            model.eval()
            for imgs, labels in valid_data_loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1) #2
                total += labels.shape[0]
                loss = F.cross_entropy(outputs,labels)
                val_loss = val_loss+loss.item()
                correct += int((predicted == labels).sum())
            val_loss = val_loss/len(valid_data_loader)
            print(val_loss ,"validation loss ------------------------------------------------")
            scheduler.step(val_loss)
    
    model.train()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset counter
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, 'best_model.pth')
        print(f"✅ Saved best model at epoch {epoch}, with validation loss {val_loss}")
    else:
        patience_counter += 1  # Increment counter
        
    # Early stopping check
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break  # Exit training loop

    
