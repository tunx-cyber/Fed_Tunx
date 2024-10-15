import data.data as data
import utils.Noniid as noniid
from utils import client
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