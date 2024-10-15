'''
 T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith,
 “Federated optimization in heterogeneous networks,” in Proc. Mach.
 Learn. Syst., 2020, pp. 429-450.
Introduce proximal term with the global model parameter via the L2 function
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from benchmark import FedBase
import random
from utils import metric, plt_figure
from utils.logger import logger
def prox_loss_fn(outputs, targets, local_weights, global_weights,mu):
    ce_loss = nn.CrossEntropyLoss()(outputs, targets)
    proximal_term = sum((lw - gw).pow(2).sum() for lw, gw in zip(local_weights, global_weights))
    prox_loss = mu * proximal_term
    return ce_loss + prox_loss
class FedProx(FedBase.FedBase):
    def __init__(self,  num_clients, model, dataset_name, mu=0.1, lr=0.01):
        super().__init__(num_clients, model, dataset_name)
        self.mu = mu
        self.lr = lr
    
    def run(self,round):
        accs = []
        random.seed(42)
        for i in range(round):
            # participants =  range(self.num_clients)
            participants = random.sample(range(0, self.num_clients), k=5)
            weights = []
            for idx in participants:
                self.clients[idx].model.load_state_dict(self.global_model.state_dict())
                w = self.clients[idx].prox_train(self.mu)
                weights.append(w)
            
            client_sizes = [self.clients[idx].train_data_size for idx in participants]
            total_size = sum(client_sizes)
            with torch.no_grad():
                for name, param in self.global_model.named_parameters():
                    weight_sum = sum(weights[i][name] *client_sizes[i]/total_size  for i in range(len(weights)))
                    param.data += weight_sum
            
            acc = metric.get_accuracy(self.clients, self.global_model)
            logger.info(f"round:[{i+1}/{round}] test_acc:{acc*100}%")
            accs.append(acc)
        
        plt_figure.draw_trainning_acc(range(round), accs, "round acc")
        self.draw_local_test()
    

