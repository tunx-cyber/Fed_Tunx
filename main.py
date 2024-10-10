'''
For test
'''
import data.data as d
import utils.client as client
import model.models as models
import torch
import unittest
import numpy as np
import benchmark.fedavg as avg
import FedGA.Fed_Tunx as fedga
class TestFed(unittest.TestCase):
    @unittest.skip("client test pass")
    def test_client(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mnist={}
        train_set = d.FedData("mnist").get_training_data()
        mnist["train"] = torch.utils.data.Subset(train_set, range(1000))

        test_set = d.FedData("mnist").get_test_data()
        mnist["test"] = torch.utils.data.Subset(test_set, range(1000))

        c = client.Client(model=models.MLP(28*28,10).to(device=device),dataset=mnist)

        state = c.train()
        print(state.keys())
        # print(state.items())
        line, col = state["fc1.weight"].size()
        print(line, col)
        print(state['fc1.weight'])
        state['fc1.weight'][0] += torch.ones(len(state['fc1.weight'][0])).to(device=device)
        print(state['fc1.weight'])
        c.test()
    
    @unittest.skip("noniid setting pass")
    def test_dirichlet(self):
        import utils.Noniid as noniid
        dist = noniid.dirichlet_setting(10, 6000)
        print(dist)

        dist = noniid.iid_setting(10, 6000)
        print(dist)

    @unittest.skip("")
    def test_fedavg(self):
        model=models.MLP(28*28,10)
        dataset_name="mnist"
        server = avg.FedAvg(num_clients=20, model=model,dataset_name=dataset_name)
        server.run()
    
    # @unittest.skip("s")
    def test_fedga(self):
        model=models.MLP(28*28,10)
        server = fedga.Fed_GA(20,model,"mnist")
        server.run(10)

if __name__ == '__main__':
    unittest.main()