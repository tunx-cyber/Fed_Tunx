import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
class Client:
    def __init__(self, model : nn.Module, dataset : dict) -> None:
        self.model = model
        self.dataset = dataset
        self.batch_size = 36
        self.size = len(dataset)
        self.epoch = 100
        self.train_data = dataset["train"]
        self.test_data = dataset["test"]

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr = 0.01)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(dataset=self.train_data, batch_size= self.batch_size, shuffle=True)

        for _ in range(self.epoch):
            for img, label in train_loader:
                optimizer.zero_grad()
                output = self.model(img)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{_+1}/{self.epoch}], Loss: {loss.item():.4f}')
        return self.model.state_dict()
    
    def test(self):

        test_loader = DataLoader(dataset=self.test_data,batch_size=self.batch_size)
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'测试准确率: {100 * correct / total:.2f}%')
            