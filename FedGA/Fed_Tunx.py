import torch.nn as nn
import utils.client as Client
import utils.Noniid as noniid
import data.data as data
import random
import copy 
import torch
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Fed_GA:
    # if the personlized is true, ouput a population with the same size
    # if the personlized is false, we only get a best performed global model
    # we first realize the global one

    def __init__(self, num_clients, model : nn.Module,dataset_name, personlized = False) -> None:
        feddata = data.FedData(dataset_name=dataset_name)

        self.train_set = feddata.get_training_data()
        self.test_set  = feddata.get_test_data()
        train_data_dist = noniid.dirichlet_setting(num_clients=num_clients, data_size=len(self.train_set))
        test_data_dist = noniid.dirichlet_setting(num_clients=num_clients, data_size=len(self.test_set))
        self.global_model = model.to(device)

        # the genes of the population, actually the parameters
        self.population = [Client.Client(model=copy.deepcopy(self.global_model).to(device),
                                dataset={ 
                                        "train": torch.utils.data.Subset(self.train_set, range(train_data_dist[i][0], train_data_dist[i][1])), 
                                        "test": torch.utils.data.Subset(self.test_set, range(test_data_dist[i][0], test_data_dist[i][1]))
                                        }, 
                                id = i ) for i in range(num_clients)] 
        
        self.personlized = personlized # need gloabl or personlized
        self.mutate_prob = 0.05
        self.num_clients = num_clients
        self.test_log = []
        self.test_g_model = copy.deepcopy(self.global_model)

    def run(self, round):
        for i in range(round):
            #Pseudoly choose available clients
            participants = random.sample(range(0, self.num_clients), k=5)

            #get the trained parameters(whole model)
            weights = []
            clients_sizes = []
            for idx in participants:
                self.population[idx].model.load_state_dict(self.global_model.state_dict())
                w = self.population[idx].train()
                clients_sizes.append(self.population[idx].data_size)
                weights.append(w)
        
            # select some parents
            self.select(weights=weights, client_sizes=clients_sizes)

    def select(self, weights, client_sizes):
        
        fitness = self.fit_eval(weights=weights, client_sizes=client_sizes)
        total = sum(fitness)
        fitness_prob = [ i/total for i in fitness]
        #demo
        father, mother =  random.choices(self.population, weights=fitness_prob,
                                           k = 2)
        return father, mother

    def crossover(self, father : nn.Module.T_destination, mother: nn.Module.T_destination):
        # we can choose a layer to exchage
        #compute how many layers
        layers = len(father.keys())
        # we can also exchange in a layer or a dimention in a layer
        # Here we can draw from a specific layer according to some rules
        layer = random.sample(range(0, layers), 1)
        layer_name = father.keys()[layer]
        # change the whole layer
        temp = father[layer_name]
        father[layer_name] = mother[layer_name]
        mother[layer_name] = temp


    # weights is the trained parameters
    def fit_eval(self, weights, client_sizes):
        # shaply value
        fitness = []
        for i in range(len(weights)):
            fitness.append(self.get_shaply_value(weights,i,client_sizes))
        return fitness
    
    def mutate(self, gene):
        # guassion mutate

        # pso mutate
        pass

    def produce(self):
        #get new offspring
        # if globla only produce one

        # if personlized produce the same size
        pass
    

    # replace global with highest individual
    def update(self):
        pass

    def get_shaply_value(self, population : list[nn.Module.T_destination], client_id, client_sizes):
        # generate a aggragation with weight?
        self.test_g_model.load_state_dict(self.global_model.state_dicit())
        shaply_value = None
        total_size = sum(client_sizes)

        with torch.no_grad:
            for name, param in self.test_g_model.named_parameters():
                weight_sum = None
                for i in range(len(population)):
                    if i != client_id:
                        weight_sum += population[i].model.state_dict()[i][name] * client_sizes[i]/total_size
                param.data = weight_sum

        shaply_value = self.model_test(self.test_g_model)
        return shaply_value

    def get_local_test_acc(self, population : list[Client.Client]):
        acc=[]
        for c in population:
            acc.append(c.test())
        return acc

    def model_test(self, model):
        test_set = self.test_set
        test_loader = DataLoader(dataset=test_set, batch_size=64)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for imgs, lables in test_loader:
                imgs, lables = imgs.to(device), lables.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += lables.size(0)
                correct += (predicted == lables).sum().item()
            
            return correct/total
