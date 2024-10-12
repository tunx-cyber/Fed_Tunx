import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader, Dataset, Subset
class Client:
    def __init__(self, model : nn.Module, dataset : dict, id = 0) -> None:
        self.model = model
        self.id = id
        self.batch_size = 32
        self.epoch = 5
        self.train_data = dataset["train"]
        self.test_data = dataset["test"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data_size = len(self.train_data)
        self.test_data_size = len(self.test_data)
        self.train_times = 0
        self.data_info = self.get_data_dist()

    def train(self):
        origin_model = copy.deepcopy(self.model)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr = 0.01)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(dataset=self.train_data, batch_size= self.batch_size, shuffle=True)

        for _ in range(self.epoch):
            #一个epoch就会遍历所有的数据，只不过一次处理的次数等于batchsize
            for img, label in train_loader:
                img, label = img.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                output = self.model(img)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
            print(f'client {self.id} Epoch [{_+1}/{self.epoch}], Loss: {loss.item():.4f}')
        
        origin_param = origin_model.state_dict()
        local_param = self.model.state_dict()

        diff = {
            key: local_param[key] - origin_param[key] for key in  origin_param.keys()
        }
        return diff
    
    def test(self):

        test_loader = DataLoader(dataset=self.test_data,batch_size=self.batch_size)
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'client {self.id} test_acc: {100 * correct / total:.2f}%')
            return correct/total
    
    def print_data_dist_info(self):
        print(f"Client {self.id} data distribution is:")
        print(self.data_info)
        print()

    def get_data_dist(self):
        train_label_dict={}
        for img, lable in self.train_data:
            if lable not in train_label_dict.keys():
                train_label_dict[lable] = 0
            train_label_dict[lable] += 1

        test_label_dict = {}
        for img, lable in self.test_data:
            if lable not in test_label_dict.keys():
                test_label_dict[lable] = 0
            test_label_dict[lable] += 1
        label_dict = {'train data':[self.train_data_size, train_label_dict],
                     'test data' : [self.test_data_size,  test_label_dict]
                    }
        return label_dict