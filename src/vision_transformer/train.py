import torch 
from vision_transformer.dataset.dataset import Plant_disease_dataset 
import sys
sys.path.append('/home/mrhouma/Documents/VisionTransformer/Vision_Transformer/src')
from vision_transformer.visiontransofrmer import visiontransformer
import torch.optim as optim
from torch.utils.data import Dataset , DataLoader , random_split , Subset
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

full_dataset = Plant_disease_dataset(
    path_data="/kaggle/input/plant-diseases/ai_training_data",
    mode='train'  
)

# Split indices
total = len(full_dataset)
train_size = int(0.85 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

indices = list(range(total))
random.shuffle(indices)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Create datasets with appropriate transforms
train_dataset_full = Plant_disease_dataset(
    path_data="/kaggle/input/plant-diseases/ai_training_data",
    mode='train'  # Training augmentation
)
val_dataset_full = Plant_disease_dataset(
    path_data="/kaggle/input/plant-diseases/ai_training_data",
    mode='val'  # Validation (no augmentation)
)

# Create subsets
train_dataset = Subset(train_dataset_full, train_indices)
valid_dataset = Subset(val_dataset_full, val_indices)
test_dataset = Subset(val_dataset_full, test_indices)

# DataLoaders with increased batch size
train_data_loader = DataLoader(
    train_dataset, 
    batch_size=32,  
    shuffle=True,
)
valid_data_loader = DataLoader(
    valid_dataset, 
    batch_size=32, 
    shuffle=False,
)

print(f"Number of classes: {len(full_dataset.disease_to_index)}")
print(f"Class mapping: {full_dataset.disease_to_index}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(valid_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Training parameters
n_epochs = 50
lr_rate = 0.0001  
patience = 5
best_val_loss = float('inf') 
patience_counter = 0  

model = visiontransformer(n_embd=256, n_classes=10).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)


from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

num_warmup_epochs = 5
num_cosine_epochs = n_epochs - num_warmup_epochs  # 45 epochs

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,      # Start at 10% of lr (0.00003)
    end_factor=1.0,        # End at 100% of lr (0.0003)
    total_iters=num_warmup_epochs
)

cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_cosine_epochs,
    eta_min=1e-6           
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[num_warmup_epochs]  
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Training loop
for epoch in range(1, n_epochs + 1): 
    start_time = datetime.datetime.now()
    loss_train = 0.0
    correct_train = 0
    total_train = 0

    model.train()
    pbar = tqdm(train_data_loader, desc=f'Epoch {epoch}/{n_epochs}')
    
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        x, labels = batch
        x, labels = x.to(device), labels.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(logits, labels,label_smoothing=0.1)
        
        # Safety check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf detected at batch {i}, epoch {epoch}!")
            continue
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
         
        
        loss_train += loss.item()
        _, predicted = torch.max(logits, dim=1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'avg_loss': f'{loss_train / (i + 1):.3f}',
            'acc': f'{100 * correct_train / total_train:.1f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    end_time = datetime.datetime.now()
    epoch_duration = (end_time - start_time).total_seconds()
    scheduler.step()
    train_acc = 100 * correct_train / total_train
    train_loss = loss_train / len(train_data_loader)
    
    if epoch == 1 or epoch % 2 == 0:
        print(f'{datetime.datetime.now()} Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Time: {epoch_duration:.2f}s')
   
    # Validation
    with torch.no_grad():
        val_loss = 0.0 
        correct = 0
        total = 0
        model.eval()
        
        # Per-class tracking
        class_correct = torch.zeros(10, device=device)  # 10 classes
        class_total = torch.zeros(10, device=device)
        
        for imgs, labels in valid_data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)
            
            # Overall metrics
            total += labels.shape[0]
            loss = F.cross_entropy(outputs, labels)
            val_loss += loss.item()
            correct += int((predicted == labels).sum())
            
            # Per-class metrics
            for label in range(10):
                mask = labels == label
                if mask.sum() > 0:
                    class_correct[label] += (predicted[mask] == labels[mask]).sum()
                    class_total[label] += mask.sum()
        
        val_loss = val_loss / len(valid_data_loader)
        val_acc = 100 * correct / total
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Validation Results:")
        print(f"{'='*60}")
        print(f"Overall Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"{'-'*60}")
        print(f"Per-Class Accuracy:")
        print(f"{'-'*60}")
        
        # Get class names from dataset
        idx_to_disease = train_dataset.dataset.index_to_disease
        
        for class_idx in range(10):
            if class_total[class_idx] > 0:
                class_acc = 100 * class_correct[class_idx] / class_total[class_idx]
                disease_name = idx_to_disease[class_idx]
                print(f"  [{class_idx}] {disease_name:30s}: {class_acc:5.1f}%  ({int(class_correct[class_idx])}/{int(class_total[class_idx])})")
            else:
                disease_name = idx_to_disease[class_idx]
                print(f"  [{class_idx}] {disease_name:30s}: No samples")
        
        print(f"{'='*60}\n")