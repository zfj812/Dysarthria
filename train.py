import torch
from torch import nn
from torch.nn import functional as F
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from utils.ConfusionMatrix import ConfusionMatrix
import datetime

def train(data_iter, net, device, batch_size = 8, test_ratio = 0.2, num_epochs = 10, learning_rate = 0.01, weight_decay = 0, tag = '2') :
    
    # net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma = 0.9)
    test_size = int(test_ratio * len(data_iter))
    train_size = len(data_iter) - test_size
    train_set, test_set = random_split(data_iter, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory= True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory= True,num_workers=4, drop_last=True)

    run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Lip_Motion_Detection_Project",
        entity="1040532700",
        name = run_name,
        group= 'autoencoder-' + str(tag),
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "transformer",
        "dataset": "LIP_DATASET",
        "epochs": num_epochs,
        "dataset_size": len(data_iter),
        "batch_size": batch_size,
        "test_ratio": test_ratio,
        }
    )
    
    criterion = nn.SmoothL1Loss(reduction='none')
    wandb.watch(net, log="all")

    for epoch in range(num_epochs+1):
        # train
        net.train()
        total_train_loss = 0
        
        for (i,(X, y)) in enumerate(train_loader):
             
            X, y = X.to(device), y.to(device)
            
            y_hat= net(X, y)

            # y_hat = y_hat[:,-1,:]
            
            mask = (y[:, :, 0] == 0).to(torch.bool)
            mask = mask.unsqueeze(-1).expand_as(y)
        
            loss = criterion(y_hat, y)
            
            loss = loss.masked_fill(mask, 0)
            valid_loss = (loss.sum() / (~mask).sum())
            optimizer.zero_grad()
            valid_loss.backward()
            optimizer.step()
            total_train_loss += valid_loss.item()
            
        scheduler.step()
        
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # wandb.log({"train_loss": avg_train_loss}, step = epoch)
        
        # evaluate
        net.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X,y in test_loader:
                X = X.to(device)
                y = y.to(device)
                out = net(X, y)
                mask = (y[:, :, 0] == 0).to(torch.bool)
                mask = mask.unsqueeze(-1).expand_as(y)
                
                # out = out[:,-1,:]
                loss = criterion(out, y) 
                loss = loss.masked_fill(mask, 0)
                valid_loss = (loss.sum() / (~mask).sum())
                
                total_val_loss += valid_loss.item()
              
        avg_val_loss = total_val_loss / len(test_loader)
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step = epoch + 1)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train MSE: {avg_train_loss:.4f}, Validation MSE: {avg_val_loss:.4f}')
        if epoch % 10 == 0:
            model_path = os.path.join(wandb.run.dir, f'model_{epoch}.pth')
            torch.save(net.state_dict(), model_path)
        
def train_classify(data_iter, net, device, batch_size = 8, test_ratio = 0.2, num_epochs = 10, learning_rate = 0.01, weight_decay = 0, tag = '2') :
    
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma = 0.8)
    test_size = int(test_ratio * len(data_iter))
    train_size = len(data_iter) - test_size
    train_set, test_set = random_split(data_iter, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory= True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory= True, drop_last=True)

    run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    confusion = ConfusionMatrix(2, ['Normal', 'Abnormal'])
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Lip_Motion_Detection_Project",
        entity="1040532700",
        name = run_name,
        group= 'classify-'+str(tag),
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "Transformer",
        "dataset": "LIP_DATASET",
        "epochs": num_epochs,
        "dataset_size": len(data_iter),
        "batch_size": batch_size,
        "test_ratio": test_ratio,
        }
    )
    
    criterion = nn.CrossEntropyLoss()
    wandb.watch(net, log="all")

    for epoch in range(num_epochs+1):
        # train
        net.train()
        total_train_loss = 0
        total_samples = 0
        trainacc = 0
        maxacc = 0
        for (i,(X, y)) in enumerate(train_loader):
             
            X, y = X.to(device), y.to(device)
            y_hat= net(X)
            loss = criterion(y_hat, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            total_train_loss += loss.item() * y.shape[0]
            total_samples += y.shape[0]
            trainacc = trainacc + (torch.argmax(y_hat,1)== y).cpu().numpy().sum()
        
        trainacc = round(trainacc / total_samples,4)
        avg_train_loss = total_train_loss / total_samples
        
        
        # evaluate
        net.eval()
        confusion.clear()

        with torch.no_grad():
            for X,y in test_loader:
                X = X.to(device)
                y = y.to(device)
                out = net(X)
                confusion.update(y.cpu().numpy(), torch.argmax(out,1).cpu().numpy())        
        confusion.summary()
        testacc = confusion.acc
        testf1 = confusion.F1
        if testacc > maxacc :
            maxacc = testacc
            confusion.plot(os.path.join(wandb.run.dir, 'confusion.png'))

        
        wandb.log({"train_loss": avg_train_loss}, step = epoch + 1)
        wandb.log({"train_acc": trainacc, "test_acc": testacc}, step = epoch + 1)
        
        
        print(f'Epoch [{epoch}/{num_epochs }], Train Loss: {avg_train_loss:.4f}, Train Acc: {trainacc:.4f}, Test Acc: {testacc:.4f}')
        if epoch % 10 == 0:
            model_path = os.path.join(wandb.run.dir, f'model_{epoch}.pth')
            torch.save(net.state_dict(), model_path)


        
if __name__ == '__main__':
    from data.dataset import LipEncodeDataset
    from models.model import lstmFcAutoEncoder
    from data.dataset import LipClassifyDataset
    from models.model import lstmClassifyModel
    from models.model import TransformerAutoEncoder
    from models.model import SimpleTransformer
    data_folder = r'E:\lyh\data\beijingdata\alldata\video_preprocessed_256\lip\normalize'

    print(data_folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(3607)
    np.random.seed(3607)
    
    dataset = LipEncodeDataset(data_folder)
    # net = lstmFcAutoEncoder(input_dim = 80, hidden_dim = 80, output_dim = 40)
    net = SimpleTransformer(80, 2, 2, 2)
    train(dataset, net, device, batch_size = 32, num_epochs = 200, learning_rate= 2e-3, tag='6', test_ratio= 0.2)
    
    # dataset = LipClassifyDataset(data_folders)
    # net1 = lstmFcAutoEncoder(input_dim = 40, hidden_dim = 40, output_dim = 20)
    # net1.load_state_dict(torch.load(r'E:\lyh\code\lip_motion_detection\wandb\run-20240507_182644-kxp4sx6h\files\model_100.pth'))
    # net = lstmClassifyModel(net1, lstm_output_dim = 20, label_num = 2)
    # net.set_encoder_trainable(True)
    # train_classify(dataset, net, device, batch_size = 1, num_epochs = 100, learning_rate= 1e-3, tag=data_folders[0][-1])
    
    wandb.finish()
        

