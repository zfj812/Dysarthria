from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
import numpy as np

class LipEncodeDataset(Dataset):
    def __init__(self, data_folder , device = 'cpu') -> None:
        super().__init__()
        
        # self.dataset = torch.randn( 100 ,100, 36).to(device)
        self.dataset = []
        self.label = []
        
        for name in os.listdir(data_folder):
            # read .npy file
            filename = os.path.join(data_folder, name)
            data = np.load(filename)

            # data = data.reshape(data.shape[0], -1)
            data = torch.tensor(data, dtype=torch.float32).to(device)
            self.dataset.append(data)
                 
            
    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.dataset[index]
        return data, label
    
    def __len__(self):
        return len(self.dataset)
    
class LipClassifyDataset(Dataset):
    def __init__(self, data_folders : list = None, label_path = '', device = 'cpu') -> None:
        super().__init__()
        
        self.dataset = []
        self.label = []
        
        if data_folders is None:
            data_folders = []
            
        for data_folder in data_folders:
            for name in os.listdir(data_folder):
                # read .npy file
                filename = os.path.join(data_folder, name)
                data = np.load(filename)
                divisor = np.array([1920, 1080])[None, None, None, :]  
                data = data / divisor
                data = data.reshape(data.shape[0], -1)
                data = torch.tensor(data, dtype=torch.float32).to(device)
                self.dataset.append(data)
                if name[0] == 'P':
                    self.label.append(1)
                else:
                    self.label.append(0)
                 
            
    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label
    
    def __len__(self):
        return len(self.dataset)
    
if __name__ == '__main__':
    data_folder = r'E:\lmj\deep\face\STAR-master\l_mouth_points_4'
    dataset = LipEncodeDataset(data_folder)

    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for (i,(X, y)) in enumerate(train_loader):
        print(X.shape)
        print(y.shape)
        