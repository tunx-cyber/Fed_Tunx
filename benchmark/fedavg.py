import utils.client as Client
import utils.Noniid as noniid
import data.data as data
import torch
import random
import copy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FedAvg:
    def __init__(self, num_clients, model, dataset_name):
        feddata = data.FedData(dataset_name=dataset_name)

        self.train_set = feddata.get_training_data()
        self.test_set  = feddata.get_test_data()
        train_data_dist = noniid.dirichlet_setting(num_clients=num_clients, data_size=len(self.train_set))
        test_data_dist = noniid.dirichlet_setting(num_clients=num_clients, data_size=len(self.test_set))
        
        self.global_model = model.to(device)
        
        self.clients = [Client.Client(model=copy.deepcopy(self.global_model).to(device),
                                dataset={ 
                                        "train": torch.utils.data.Subset(self.train_set, range(train_data_dist[i][0], train_data_dist[i][1])), 
                                        "test": torch.utils.data.Subset(self.test_set, range(test_data_dist[i][0], test_data_dist[i][1]))
                                        }, 
                                id = i ) for i in range(num_clients)]
        self.num_clients = num_clients
        
        self.test_log = []

    #self.model.state_dict() 可以使用key()还有items()
    def run(self, round = 10):
        for i in range(round):
            participants =  random.sample(range(0, self.num_clients), 2)
            weights = []
            for idx in participants:
                self.clients[idx].model.load_state_dict(self.global_model.state_dict())
                w = self.clients[idx].train()
                weights.append(w)
            
            client_sizes = [self.clients[idx].data_size for idx in participants]
            total_size = sum(client_sizes)
            with torch.no_grad():
                for name, param in self.global_model.named_parameters():
                    weight_sum = sum(weights[i][name] *client_sizes[i]/total_size  for i in range(len(weights)))
                    param.data += weight_sum
            
            for idx in participants:
                self.clients[idx].model.load_state_dict(self.global_model.state_dict())
            self.test()
        
        plt.plot(range(round), self.test_log, label='test_acc', color='blue')
        plt.title('test_acc')
        plt.xlabel('round')
        plt.ylabel('acc')
        plt.show()
        
    
    def test(self):
        test_set = self.test_set
        test_loader = DataLoader(dataset=test_set,batch_size=64)
        self.global_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'global_test_acc: {100 * correct / total:.2f}%')
            self.test_log.append(100 * correct / total)

