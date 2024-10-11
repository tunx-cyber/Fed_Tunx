import torch.nn as nn
import utils.client as Client
import utils.Noniid as noniid
import data.data as data
import random
import copy 
import torch
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
from utils import timer
'''
fitness evaluation
select
crossover get new offsprings
mutate on offsprings
'''
class Fed_GA:
    # if the personlized is true, ouput a population with the same size
    # if the personlized is false, we only get a best performed global model
    # we first realize the global one

    def __init__(self, num_clients, model : nn.Module,dataset_name, personlized = False) -> None:
        feddata = data.FedData(dataset_name=dataset_name)

        self.train_set = feddata.get_training_data()
        self.test_set  = feddata.get_test_data()
        train_data_dist = noniid.dirichlet_setting(num_clients=num_clients,dataset=self.train_set)
        test_data_dist = noniid.dirichlet_setting(num_clients=num_clients,dataset=self.test_set)
        self.global_model = model.to(device)

        # the genes of the population, actually the parameters
        self.population = [Client.Client(model=copy.deepcopy(self.global_model).to(device),
                                dataset={ 
                                        "train": train_data_dist[i], 
                                        "test": test_data_dist[i]
                                        }, 
                                id = i ) for i in range(num_clients)] 
        
        self.personlized = personlized # need gloabl or personlized
        self.mutate_prob = 0.3
        self.num_clients = num_clients
        self.test_log = []
        self.test_g_model = copy.deepcopy(self.global_model)

        self.small_test_set = noniid.get_iid_set(self.test_set,10)

    def local_training(self, participants):
        #get the trained parameters（difference)delta
        weights = []
        clients_sizes = []
        for idx in participants:
            self.population[idx].model.load_state_dict(self.global_model.state_dict())
            w = copy.deepcopy(self.population[idx].train())
            clients_sizes.append(self.population[idx].data_size)
            weights.append(w)
        
        return weights, clients_sizes
            
    #TODO:对GA部分并行化 
    def run(self, round=10):
        accs = []
        for r in range(round):
            #Pseudoly choose available clients
            participants = random.sample(range(0, self.num_clients), k=5)
            # participants = range(self.num_clients)
            for i in participants:
                self.population[i].train_times+=1
            #get the trained parameters（difference)delta
            weights, clients_sizes = self.local_training(participants)
            
            #get the fitnesses
            fitness = self.fit_eval(weights=weights, client_sizes=clients_sizes)
            
            #clone a best model
            index = fitness.index(max(fitness))
            best_weight = copy.deepcopy(weights[index])
            # select some parents 找最相近的交换？而不是最优的几个？
            for i in range(len(participants)):
                father, mother = self.select(weights=weights,fitness_prob=fitness)
                #father will be a new father, so does mother
                self.crossover(father=father,mother= copy.deepcopy(best_weight))
                self.crossover(father=mother,mother=copy.deepcopy(best_weight))
            
            #new model has been updated, we are going to mutate
            self.mutate(best_weight=best_weight,weights=weights)
                     
            #we can top k
            self.aggregate(weights=weights,client_sizes=clients_sizes)
            
            acc = self.model_test(self.global_model)
            accs.append(acc)
            print(f"round:[{r+1}/{round}] test_acc:{acc*100}%")

        self.draw_acc(accs,range(round),"round acc")
        accs = self.get_local_test_acc(self.population)
        self.draw_acc(accs,range(self.num_clients),"client acc")

        for c in self.population:
            print(c.train_times)

    def draw_acc(self,accs,x,s):
        plt.plot(x, accs, label='test_acc', color='blue')
        plt.title(s)
        plt.xlabel('x')
        plt.ylabel('acc')
        plt.show()

    def select(self,weights, fitness_prob):
        #demo
        father, mother =  random.choices(weights, weights=fitness_prob,
                                           k = 2)
        return father, mother

    def crossover(self, father : nn.Module.T_destination, mother: nn.Module.T_destination):
        # we can choose a layer to exchage
        # we can also exchange in a layer or a dimention in a layer
        # Here we can draw from a specific layer according to some rules
        layer_name = random.sample(list(father.keys()), 1)[0]
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
        fitness = self.normalize(fitness)
        total = sum(fitness)
        fitness_prob = [ i/total for i in fitness]
        return fitness_prob
    
    def mutate(self, best_weight,weights):
        for i in range(len(weights)):
            if random.random() < self.mutate_prob:
                self.pso_mutate(best_gene=best_weight, gene=weights[i])
    
    @timer.timer
    def get_shaply_value(self, population : list[nn.Module.T_destination], client_id, client_sizes):
        # generate a aggragation with weight?
        self.test_g_model.load_state_dict(self.global_model.state_dict())
        total_size = sum(client_sizes)

        #这里可以根据name分割 并行计算
        with torch.no_grad():
            for name, param in self.test_g_model.named_parameters():
                weight_sum = torch.zeros(param.data.size()).to(device=device)
                for i in range(len(population)):
                    if i != client_id:
                        weight_sum += population[i][name] * client_sizes[i]/total_size
                param.data += weight_sum

        shaply_value_without = self.model_test(self.test_g_model)
        with torch.no_grad():
            for name, param in self.test_g_model.named_parameters():
                param.data+=population[client_id][name] * client_sizes[client_id]/total_size

        shaply_value_with = self.model_test(self.test_g_model)
        shaply_value = shaply_value_with - shaply_value_without
        return shaply_value

    def get_local_test_acc(self, population : list[Client.Client]):
        acc=[]
        for c in population:
            c.model.load_state_dict(self.global_model.state_dict())
            acc.append(c.test())
        return acc

    def model_test(self, model):
        test_set = self.small_test_set
        test_loader = DataLoader(dataset=test_set, batch_size=10)
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
    
    def draw_mean_stddev(self, clients):
        pass
    #TODO: acctually we can study how to determine the mean and stddev
    #one strategy is that we can draw from from the uploaded clients
    #shall we mutate on one parameter or a layer or whole
    #or we choose the most important one
    def gaussion_mutate(self, gene : nn.Module.T_destination, mean, stddev):
        for key in gene.keys():
            gene[key]+=torch.normal(mean, stddev,size=gene[key].size()).to(device=device)
    
    '''
    v = w * v + c1 * r1*(best - x) + c2 * r2 * (best-x)
    x = x + v
    v:当前速度
    x:粒子位置
    best: 最佳位置
    w: 惯性权重
    c1 c2: 学习因子
    r1 r2: 0-1 随机数
    收到这个启发，我们考虑梯度的方向，尽可能让所有参数向最优方向移动即可
    '''
    def pso_mutate(self, best_gene, gene):
        c1 = 1.5
        c2 = 1.5
        r1 = random.random()
        r2 = random.random()
        #v可以通过某种贪心策略计算出来这里我们不考虑
        for key in gene.keys():
            gene[key]+= c1*r1*best_gene[key]

    def aggregate(self, weights,client_sizes):
        total_size = sum(client_sizes)
        with torch.no_grad():
            for key in self.global_model.state_dict().keys():
                for i in range(len(weights)):
                    self.global_model.state_dict()[key] += weights[i][key]*client_sizes[i]/total_size
    
    def normalize(self,data):
        min_val = min(data)
        max_val = max(data)
        return [(x - min_val) / (max_val - min_val) for x in data]