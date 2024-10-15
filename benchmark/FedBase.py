import torch.nn as nn
import data.data as data
import utils.Noniid as noniid
from utils import client,plt_figure
import torch
from torch.utils.data import DataLoader
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FedBase:
    def __init__(self, num_clients, model, dataset_name, alpha=None):
        feddata = data.FedData(dataset_name=dataset_name)
        self.train_set = feddata.get_training_data()
        self.test_set  = feddata.get_test_data()
        train_data_dist = noniid.dirichlet_setting(num_clients=num_clients,dataset=self.train_set,alpha=alpha)
        test_data_dist = noniid.dirichlet_setting(num_clients=num_clients,dataset=self.test_set,alpha=alpha)
        self.global_model = model.to(device)
        self.label_size = len(set(self.test_set.targets))
        
        self.clients = [client.Client(model=copy.deepcopy(self.global_model).to(device),
                                dataset={ 
                                        "train": train_data_dist[i], 
                                        "test": test_data_dist[i]
                                        }, 
                                label_size = self.label_size,
                                id = i ) for i in range(num_clients)]
        self.num_clients = num_clients
        self.small_test_set = noniid.get_iid_set(self.test_set,20)
    

    # public data test
    def test(self):
        test_set = self.small_test_set
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
            return correct, total
    
    def draw_local_test(self):
        accs = self.get_local_test_acc(self.clients)
        plt_figure.draw_clients_acc_bar(self.num_clients, accs)
    
    def get_local_test_acc(self, clients : list[client.Client]):
        acc=[]
        for c in clients:
            c.model.load_state_dict(self.global_model.state_dict())
            corr, tol = c.test()
            acc.append(corr/tol)
        return acc
    
        # public data test  
    def model_test(self, model):
        criterion = nn.CrossEntropyLoss() 
        test_set = self.small_test_set
        test_loader = DataLoader(dataset=test_set, batch_size=10)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0.0
            for imgs, lables in test_loader:
                imgs, lables = imgs.to(device), lables.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == lables).sum().item()
                total += lables.size(0)
                total_loss += criterion(outputs, lables).item()*imgs.size(0)
            
            return correct, total, total_loss
        
    def local_training(self, participants):
        #get the trained parameters（difference)delta
        weights = []
        clients_sizes = []
        for idx in participants:
            self.clients[idx].model.load_state_dict(self.global_model.state_dict())
            w = copy.deepcopy(self.clients[idx].train())
            clients_sizes.append(self.clients[idx].train_data_size)
            weights.append(w)
        
        return weights, clients_sizes
    
    #base aggregate method
    def aggregate(self, weights,client_sizes):
            total_size = sum(client_sizes)
            with torch.no_grad():
                for key in self.global_model.state_dict().keys():
                    for i in range(len(weights)):
                        self.global_model.state_dict()[key] +=self.lr * weights[i][key]*client_sizes[i]/total_size