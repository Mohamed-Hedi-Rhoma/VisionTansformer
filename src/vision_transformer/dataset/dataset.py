import os 
import torch
from torch.utils.data import Dataset , DataLoader , random_split
from PIL import Image 
from torchvision import transforms



class Plant_disease_dataset(Dataset):
    def __init__(self , path_data ):
        super().__init__()
        self.path_data = path_data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),           # Convert to tensor [0, 1]
            transforms.Normalize(mean=[0.4728, 0.4893, 0.3695],  # ImageNet stats
                                std=[0.2178, 0.2103, 0.2195])
        ])
        self.images_path = []
        self.labels = []
        self.disease_to_index = {}
        self.index_to_disease = {}
        for  i , subdir in enumerate(os.listdir(path_data)) : 
            subdir_path = os.path.join(path_data , subdir)
            if os.path.isdir(subdir_path) : 
                disease_name =  os.path.basename(subdir_path)
                self.disease_to_index[disease_name] = i 
                self.index_to_disease[i] = disease_name
                for file_name in os.listdir(subdir_path) :
                    file_name_path = os.path.join(subdir_path,file_name)
                    if os.path.isfile(file_name_path) and file_name_path.endswith(('.jpg', '.jpeg', '.png')) : 
                        print(f"Adding file path{file_name_path}")
                        self.images_path.append(file_name_path)
                        print(f"Adding label {i}")
                        self.labels.append(i)
        
    def __len__(self) : 
        return(len(self.images_path))
    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image ,label


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