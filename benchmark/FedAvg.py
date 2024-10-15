'''
 B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. Y. Arcas,
 “Communication-efficient learning of deep networks from decentralized
 data,” in Proc. Int. Conf. Artif. Intell. Statist., 2017, pp. 1273-1282.
'''
import torch
from benchmark import FedBase
import random
from utils import metric, plt_figure
from utils.logger import logger
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FedAvg(FedBase.FedBase):
    def __init__(self, num_clients, model, dataset_name):
        super().__init__(num_clients, model, dataset_name)
            
    #self.model.state_dict() 可以使用key()还有items()
    def run(self, round = 10):
        accs = []
        random.seed(42)
        for i in range(round):
            # participants =  range(self.num_clients)
            participants = random.sample(range(0, self.num_clients), k=5)
            weights = []
            for idx in participants:
                self.clients[idx].model.load_state_dict(self.global_model.state_dict())
                w = self.clients[idx].train()
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
        