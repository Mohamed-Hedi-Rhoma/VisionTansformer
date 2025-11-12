import os 
import torch
from torch.utils.data import Dataset , DataLoader , random_split
from PIL import Image 
from torchvision import transforms



class Plant_disease_dataset(Dataset):
    def __init__(self, path_data, mode='train'):
        super().__init__()
        self.path_data = path_data
        self.mode = mode
        
        # Different transforms for train vs validation
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(288),  # Resize larger first
                transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # Random crop to 256x256
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),  # Plants can be rotated
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),  # Random translation
                    scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4667, 0.4933, 0.3419],
                    std=[0.2139, 0.2060, 0.2118]
                )
            ])
        else:  # validation/test
            self.transform = transforms.Compose([
                transforms.Resize(288),
                transforms.CenterCrop(256),  # Center crop to 256x256
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4667, 0.4933, 0.3419],
                    std=[0.2139, 0.2060, 0.2118]
                )
            ])
        
        self.images_path = []
        self.labels = []
        self.disease_to_index = {}
        self.index_to_disease = {}
        
        for i, subdir in enumerate(sorted(os.listdir(path_data))):  # Sort for consistency
            subdir_path = os.path.join(path_data, subdir)
            if os.path.isdir(subdir_path): 
                disease_name = os.path.basename(subdir_path)
                self.disease_to_index[disease_name] = i 
                self.index_to_disease[i] = disease_name
                
                for file_name in sorted(os.listdir(subdir_path)):
                    file_name_path = os.path.join(subdir_path, file_name)
                    if os.path.isfile(file_name_path) and file_name_path.lower().endswith(('.jpg', '.jpeg', '.png')): 
                        self.images_path.append(file_name_path)
                        self.labels.append(i)
        
        print(f"Loaded {len(self.images_path)} images in {mode} mode")
    
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label


"""
data =Plant_disease_dataset(path_data="/home/mrhouma/Documents/VisionTransformer/Vision_Transformer/ai_training_data")
total = len(data)
train_size = int(0.8 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size
train_dataset, valid_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])
train_data_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)
for sample in train_data_loader :
    imgs ,labels= sample
    print(imgs.shape , labels.shape) 
print(data[0])"""