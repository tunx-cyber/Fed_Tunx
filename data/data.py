import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

mnist_transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))])

class FedData:
    def __init__(self, dataset_name:str) -> None:
        self.root_name = "dataset/"+dataset_name
        self.dataset_name = dataset_name
        pass

    def get_data(self, train:bool):
        data = None
        #TODO: check if we need to download
        download = True
        if self.dataset_name.lower() == "mnist" :
            data = datasets.MNIST(
                root=self.root_name,
                train=train,
                download=download,
                transform=mnist_transform
            )
        elif self.dataset_name.lower() == "emnist" : 
            data = datasets.EMNIST(
                root=self.root_name,
                train=train,
                download=True,
                transform=ToTensor()
            )
        elif self.dataset_name.lower() == "cifar10" :
            data = datasets.CIFAR10(
                root=self.root_name,
                train=train,
                download=True,
                transform=ToTensor()
            )
        elif self.dataset_name.lower() == "cifar10" :
            data = datasets.CIFAR100(
                root=self.root_name,
                train=train,
                download=True,
                transform=ToTensor()
            )
        else:
            print("Invalid dataset name")

        return data
    
    def get_training_data(self):
        training_data = self.get_data(True)
        return training_data

    def get_test_data(self):
        test_data = self.get_data(False)
        return test_data

    #TODO
    def get_iid_data(self):
        pass
    
    #TODO
    def get_nonidd_data(self):
        pass

