import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
class Client:
    def __init__(self, model : nn.Module, dataset) -> None:
        self.model = model
        self.dataset = dataset
        self.batch_size = 36
        self.size = len(dataset)
        self.epoch = 100

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr = 0.01)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(dataset=self.dataset, batch_size= self.batch_size, shuffle=True)

        for _ in range(self.epoch):
            for img, label in train_loader:
                optimizer.zero_grad()
                output = self.model(img)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
        
        return self.model.state_dict()
    
    def local_test(self):
        pass