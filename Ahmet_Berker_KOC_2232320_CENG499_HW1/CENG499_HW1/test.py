import torch 
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset_test import MnistDataset1
from model import MyModel
import torch.nn.functional as F


def test(model,test_dataloader,device,test_dataset):
    model.load_state_dict(torch.load('model_folder\model_state_dict89'))         
    model.eval()
    with torch.no_grad():
            
        
        counter=0
        file1 = open("test_label.txt","a")
        for images,labels in test_dataloader:
                
            images = images.to(device)
            pred = model(images,1,2)
            pred_list=torch.max(pred,1)
            pred_index=pred_list[1]
            for i in range(pred_index.size(0)):
                x=test_dataset.data[i+64*counter][0].split('\\')[2]
                file1.write('{} {}\n'.format(x, pred_index[i].item()))
            counter=counter+1
        file1.close
            


def main():
    use_cuda = True #GPU 
    device = torch.device('cuda' if  use_cuda else 'cpu') #set device gpu or cpu just changing use_cuda
    epochs = 40 #epoch number
    torch.manual_seed(123)
    
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, ), (0.5, )),
    ])
     
    test_dataset = MnistDataset1('data','test', transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=4)
    
    model = MyModel(512)
    model = model.to(device)
    test(model,test_dataloader,device,test_dataset)
    
                    
    
if __name__ == '__main__':
    
    main()