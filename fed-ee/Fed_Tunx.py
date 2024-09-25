import torch.nn as nn
import random
import client
class Fed_GA:
    # if the personlized is true, ouput a population with the same size
    # if the personlized is false, we only get a best performed global model
    # we first realize the global one

    def __init__(self, population, personlized = False) -> None:
        self.population : list[nn.Module.T_destination] = population # the genes of the population, actually the parameters
        self.personlized = personlized # need gloabl or personlized
        self.size = len(population)# the size of population
        self.mutate_prob = 0.05

        pass
    
    def select(self):
        fitness = self.fit_eval()
        total = sum(fitness)
        fitness_prob = [ i/total for i in fitness]
        
        #demo
        father, mother =  random.choices(self.population, weights=fitness_prob,
                                           k = 2)

        return father, mother

    def crossover(self, father : nn.Module.T_destination, mother: nn.Module.T_destination):
        # we can choose a layer to exchage
        #TODO
        father.keys()
        # we can also exchange in a layer

    def fit_eval(self):
        # shaply value
        fitness = []

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

    def get_shaply_value(self):
        # generate a aggragation with weight?
        weights = None
        weights = [1.0 for i in range(self.size)]

