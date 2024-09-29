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
class TestFed(unittest.TestCase):
    @unittest.skip("clinet test pass")
    def test_client(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mnist={}
        train_set = d.FedData("mnist").get_training_data()
        mnist["train"] = torch.utils.data.Subset(train_set, range(1000))

        test_set = d.FedData("mnist").get_test_data()
        mnist["test"] = torch.utils.data.Subset(test_set, range(1000))

        c = client.Client(model=models.MLP(28*28,10).to(device=device),dataset=mnist)

        c.train()
        c.test()
    
    @unittest.skip("noniid setting pass")
    def test_dirichlet(self):
        import utils.Noniid as noniid
        dist = noniid.dirichlet_setting(10, 6000)
        print(dist)

        dist = noniid.iid_setting(10, 6000)
        print(dist)

    @unittest.skip("fedavg setting pass")
    def test_fedavg(self):
        model=models.MLP(28*28,10)
        dataset_name="mnist"
        server = avg.FedAvg(num_clients=20, model=model,dataset_name=dataset_name)
        server.run()

if __name__ == '__main__':
    unittest.main()