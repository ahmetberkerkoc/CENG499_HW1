import torch 
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import MnistDataset
from model import MyModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
def train(model, optimizer, train_dataloader,validation_dataloader, epochs, device,model_number,activation,layer_size):
    best_val_loss=float('inf')
    model_val_accuracy=0
    model.train()
    counter=0
    train_plot=[]
    valid_plot=[]
    listt=[]
    for epoch_idx in range(epochs): #training loop
        print('number of epoch {}'.format(epoch_idx))
        total_train_loss=0
        train_acc=0
        for images, labels in train_dataloader: #for whole dataset 
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images,activation,layer_size)
            pred_list=torch.max(pred,1)
            pred_index=pred_list[1]
            loss = F.nll_loss(pred, labels)
            loss.backward()
            optimizer.step()
            total_train_loss=total_train_loss+loss
            train_acc += torch.sum(pred_index == labels)
            #train_correct += (pred == labels).float().sum()
        avarage_train_loss=total_train_loss/len(train_dataloader)
        train_plot.append(avarage_train_loss.item())
        print('Train Loss: {}'.format(avarage_train_loss.item()))
        train_accuracy_total=(100*train_acc)/25000
        print('Train Accuracy: {}'.format(train_accuracy_total))
        model.eval()
        with torch.no_grad():
            
            total_val_loss=0
            val_acc=0
            for images, labels in validation_dataloader:
                
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images,activation,layer_size)
                pred_list=torch.max(pred,1)
                pred_index=pred_list[1]
                val_loss = F.nll_loss(pred, labels)
                total_val_loss=total_val_loss+val_loss
                val_acc += torch.sum(pred_index == labels)
                #val_correct += (pred == labels).float().sum()
    
        avarage_val_loss=total_val_loss/len(validation_dataloader)
        valid_plot.append(avarage_val_loss.item())
        print('Validation Loss: {}'.format(avarage_val_loss.item()))
        total_val_accuracy=(100*val_acc)/5000
        print('Validation Accuracy: {}'.format(total_val_accuracy))
        
        if avarage_val_loss.item() < best_val_loss:
            #print('Last Validation Loss: {}'.format(best_val_loss))
            counter = 0 
            print('model is saved for epoch {}'.format(epoch_idx))
            torch.save(model.state_dict(), 'model_folder\model_state_dict{}'.format(model_number))
            best_val_loss=avarage_val_loss.item()
            model_val_accuracy = total_val_accuracy
    file1 = open("acc_loss.txt","a")
    file1.write('loss: {} and acc: {} for model_number: {} \n'.format(best_val_loss, model_val_accuracy,model_number))
    file1.close
    listt.append(train_plot)
    listt.append(valid_plot)
    return listt
'''    
        else:
            print('VALIDATION LOSS NOT DECREASE !!! and model not saved')
            counter = counter +1
            if counter>14:
                print('Training is stop in epoch{} '.format(epoch_idx))
'''       
def main():
    use_cuda = True #GPU 
    device = torch.device('cuda' if  use_cuda else 'cpu') #set device gpu or cpu just changing use_cuda
    epochs = 60 #epoch number
    plot_val=[]
    plot_train=[]
    listt=[]
    torch.manual_seed(123)
    
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, ), (0.5, )),
    ])
     
    dataset = MnistDataset('data','train', transforms)
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset,[25000, 5000])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True,num_workers=4)
    model_number=1000
    learning_rate=[0.01, 0.03,0.001, 0.003, 0.0001, 0.0003]
    neuron_size=[256, 512, 1024]
    for layer_size in range(2,3):
        for activation in range(1,2):
            for neuron in range (1,2):
                for lrr in range(4,5): #learning rate
                    model = MyModel(neuron_size[neuron])
                    
                    model = model.to(device)
                    model_number = model_number + 1
                    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate[lrr]) #learning rate
                    listt=train(model, optimizer, train_dataloader, validation_dataloader, epochs, device,model_number,activation,layer_size)
                    plt.plot(listt[0])
                    plt.plot(listt[1])
                    plt.legend(["Train_Loss", "Val_Loss"])
                    plt.xlabel('Number of Epoch')
                    plt.ylabel('Train and Validation Losses')
                    plt.show()
if __name__ == '__main__':
    
    main()