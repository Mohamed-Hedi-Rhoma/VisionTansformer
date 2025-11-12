from PIL import Image 
import os 
import torch 
from torchvision import transforms

width_total = 0
height_total = 0 
path_data = "/home/mrhouma/Documents/VisionTransformer/ai_training_data"
nb = 0
for dir in os.listdir(path_data) :
    subdir = os.path.join(path_data,dir)
    if os.path.isdir(subdir) : 
        for file in os.listdir(subdir) : 
            file_path = os.path.join(subdir,file)
            if os.path.isfile(file_path) and file_path.endswith(('.jpg', '.jpeg', '.png')) : 
                image = Image.open(file_path)
                width , height = image.size 
                print(f"width {file_path,width}")
                print(f"height {file_path,height}")
                width_total+= width
                height_total+= height
                nb +=1 
print(f"Mean width : ------ {width_total/nb}")
print(f"Mean height : ------ {height_total/nb}")
                

L=[]

for dir in os.listdir(path_data) :
    subdir = os.path.join(path_data,dir)
    if os.path.isdir(subdir) : 
        for file in os.listdir(subdir) : 
            file_path = os.path.join(subdir,file)
            if os.path.isfile(file_path) and file_path.endswith(('.jpg', '.jpeg', '.png')) : 
                image = Image.open(file_path).convert('RGB')
                image = transforms.Resize((224, 224))(image)  # Resize PIL image
                image_tensor = transforms.ToTensor()(image)   # Then to tensor
                L.append(image_tensor)

imgs = torch.stack(L,dim=3)
print(imgs.shape)
print(imgs.view(3,-1).shape)
print(imgs.view(3,-1).mean(dim=1))
print(imgs.view(3, -1).std(dim=1))
                

                
