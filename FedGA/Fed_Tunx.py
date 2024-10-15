import torch.nn as nn
import utils.client as Client
import utils.Noniid as noniid
import data.data as data
import random
import copy 
import torch
from torch.utils.data import DataLoader
from utils import timer, plt_figure
from utils.logger import logger
import numpy as np
from utils import metric
from benchmark.FedBase import FedBase
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
fitness evaluation
select
crossover get new offsprings
mutate on offsprings
'''
class Fed_GA(FedBase):
    # if the personlized is true, ouput a population with the same size
    # if the personlized is false, we only get a best performed global model
    # we first realize the global one

    def __init__(self, num_clients, model : nn.Module,dataset_name, personlized = False) -> None:
        super().__init__(num_clients=num_clients,model=model,dataset_name=dataset_name)

        labels = np.array(self.test_set.targets)
        self.label_size = len(set(labels))

        # the genes of the population, actually the parameters
        self.population = self.clients
        
        self.personlized = personlized # need gloabl or personlized
        self.mutate_prob = 0.3
        self.num_clients = num_clients
        self.test_g_model = copy.deepcopy(self.global_model)
        self.lr = 1
            
     
    def run(self, round=10):
        accs = []
        random.seed(42)
        for r in range(round):
            #Pseudoly choose available clients
            #TODO the clients choose strategy based on the gradient direction

            # participants are the indices of clients
            participants = random.sample(range(0, self.num_clients), k=5)
            # participants = range(self.num_clients)
            for i in participants:
                self.population[i].train_times+=1

            data_dist_vecs = []
            for i in participants:
                data_dist_vecs.append(self.population[i].data_info['data_dist_vec'])
            
            # logger.info(f'sim matrix \n{np.array(self.get_similarity_matrix(data_dist_vecs))}')
            
            #get the trained parameters（difference)delta
            weights, clients_sizes = self.local_training(participants)
            
            #TODO:对GA部分并行化
            for i in range(10):     
                self.New_GA(weights=weights, clients_sizes=clients_sizes, participants=participants)
                     
            #we can top k
            self.aggregate(weights=weights,client_sizes=clients_sizes)
            
            acc = metric.get_accuracy(self.population, self.global_model)
            accs.append(acc)
            # print(f"round:[{r+1}/{round}] test_acc:{acc*100}%")
            logger.info(f"round:[{r+1}/{round}] test_acc:{acc*100}%")

        self.draw_acc(range(round), accs, "round acc")
        self.draw_local_test()
        for client in self.population:
            logger.info(f'Client [{client.id} trained times: {client.train_times}]')

    def draw_acc(self,rounds, accs, title):
        plt_figure.draw_trainning_acc(rounds=rounds, accs=accs, title=title)

    def select(self,weights, fitness_prob):
        #demo
        father, mother =  random.choices(weights, weights=fitness_prob,
                                           k = 2)
        return father, mother

    def crossover(self, father : nn.Module.T_destination, mother: nn.Module.T_destination):
        # we can choose a layer to exchage
        # we can also exchange in a layer or a dimention in a layer
        # Here we can draw from a specific layer according to some rules
        # layer_name = random.sample(list(father.keys()), 1)[0]
        # layer_name = list(father.keys())[-2]
        # # change the whole layer
        # temp = father[layer_name]
        # father[layer_name] = mother[layer_name]
        # mother[layer_name] = temp

        # layer_name = list(father.keys())[-1]
        # father[layer_name], mother[layer_name] = mother[layer_name], father[layer_name]
        child = {}
        for key in father.keys():
            if np.random.rand() < 0.2:
                child[key] = father[key]
            else:
                child[key] = mother[key]
        return child

    # weights is the trained parameters
    def fit_eval(self, weights, client_sizes):
        # shaply value
        fitness = []
        for i in range(len(weights)):
            fitness.append(self.get_shaply_value(weights,i,client_sizes))
        fitness = self.normalize(fitness)
        total = sum(fitness)

        fitness_prob = None
        if total == 0:
            fitness_prob = [1/len(fitness) for i in range(len(fitness))]
        else:
            fitness_prob = [ i/total for i in fitness]
        return fitness_prob
    
    def mutate(self, best_weight,weights):
        for i in range(len(weights)):
            if random.random() < self.mutate_prob:
                self.pso_mutate(best_gene=best_weight, gene=weights[i])
    
    # @timer.timer
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
        c1 = 1.1
        r1 = random.random()
        #v可以通过某种贪心策略计算出来这里我们不考虑
        for key in gene.keys():
            if random.random() < self.mutate_prob:
                gene[key]+= c1*r1*best_gene[key]
    
    def normalize(self,data):
        min_val = min(data)
        max_val = max(data)
        if min_val == max_val:
            return [1/len(data) for _ in data]
        return [(x - min_val) / (max_val - min_val) for x in data]

    # similarity based on data distribution
    def get_similarity_matrix(self, data_dist_vecs):
        matrix = []
        for v1 in data_dist_vecs:
            sim = []
            for v2 in data_dist_vecs:
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                sim.append(cos_sim)
            matrix.append(sim)
        
        return matrix    

    def GA(self, weights, clients_sizes, participants):
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

    def New_GA(self,weights, clients_sizes,participants ):
        fitness=[]
        for index in range(len(weights)):
            self.test_g_model.load_state_dict(self.global_model.state_dict())
            for name, param in self.test_g_model.named_parameters():
                param.data += weights[index][name] #先不考虑大小的问题
            test_res = self.model_test(self.test_g_model)
            fitness.append(-test_res[2]/test_res[1])
        
        #归一化
        fitness = self.normalize(fitness)

        #保证和为一
        t = sum(fitness)
        fitness = [x/t for x in fitness]

        best_weight =copy.deepcopy(weights[np.argmax(fitness)])
        #选择父母
        offsprings = []
        for _ in range(len(weights)//2):
            p1, p2 = np.random.choice(range(len(participants)), size=2,
            p=fitness)
            child1 = self.crossover(weights[p1], weights[p2])
            child2 = self.crossover(weights[p1], weights[p2])
            # self.pso_mutate(best_gene=best_weight,gene=child1)
            # self.pso_mutate(best_gene=best_weight,gene=child2)
            offsprings.append(child1)
            offsprings.append(child2)
        
        fitness=[]
        for w in offsprings:
            self.test_g_model.load_state_dict(self.global_model.state_dict())
            for name, param in self.test_g_model.named_parameters():
                param.data += w[name] #先不考虑大小的问题
            test_res = self.model_test(self.test_g_model)
            fitness.append(-test_res[2]/test_res[1])
        
        arr=[[k,fitness[k]] for k in range(len(fitness))]
        sorted_array = sorted(arr, key=lambda x: x[1],reverse=True)
        #把weights后面的换成新的
        for i in range(len(offsprings)//2):
            weights[-(i+1)] = offsprings[sorted_array[i][0]]
