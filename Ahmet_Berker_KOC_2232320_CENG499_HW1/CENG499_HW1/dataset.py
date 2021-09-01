import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T 

class MnistDataset(Dataset):
    def __init__(self, dataset_path, split, transforms):
        images_path = os.path.join(dataset_path, split) #concatanete path (path of all images)
        self.data = []
        with open(os.path.join(images_path, 'labels.txt'),'r') as f:
            for line in f: 
                image_name, label = line.split() #if we split the line
                image_path = os.path.join(images_path, image_name) #concatanation of image path and image name (path of image)
                label = int(label)
                self.data.append((image_path, label))
        self.transforms = transforms

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path = self.data[index][0] #take image path
        label=self.data[index][1] #take label
        image =Image.open(image_path)
        image = self.transforms(image)
        return image, label


if __name__ == '__main__':
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, ), (0.5, )),
    ])
    dataset=MnistDataset('data','train', transforms)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,num_workers=4)
    for images, labels in dataloader:
        print(images.size())
        print(labels)
        exit()
    print(len(dataset))
    #print(dataset[0])