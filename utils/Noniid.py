import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, Subset
import random


# 定义一个自定义数据集类
class NonIIDMNIST(Dataset):
    def __init__(self, dataset, num_clients, alpha):
        self.data = dataset
        self.num_clients = num_clients
        self.alpha = alpha
        self.client_data_indices = self.generate_non_iid_indices()

    def generate_non_iid_indices(self):
        # 获取所有标签
        y = np.array(self.data.targets)
        client_indices = [[] for _ in range(self.num_clients)]
        
        # 从迪列克雷分布中生成每个客户端的类别比例
        proportions = np.random.dirichlet(self.alpha, self.num_clients)

        # 为每个客户端选择样本
        for client in range(self.num_clients):
            client_labels = []
            for digit in range(10):
                num_samples = int(proportions[client][digit] * len(y[y == digit]) * 0.8)  # 80% 的样本
                indices = np.where(y == digit)[0]
                selected_indices = np.random.choice(indices, num_samples, replace=False)
                client_indices[client].extend(selected_indices)
                client_labels.extend([digit] * num_samples)

            # 添加剩余样本（如有）
            remaining_indices = np.setdiff1d(np.arange(len(y)), client_indices[client])
            client_indices[client].extend(remaining_indices.tolist())

        return client_indices

    def __len__(self):
        return len(self.client_data_indices)

    def __getitem__(self, idx):
        client_idx = idx % self.num_clients
        sample_indices = self.client_data_indices[client_idx]
        return self.data[sample_indices]

def main():
    # 加载 MNIST 数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    num_clients = 5
    alpha = [1.0] * 10  # 迪列克雷分布参数
    non_iid_mnist = NonIIDMNIST(mnist_dataset, num_clients, alpha)

    # 创建客户端数据加载器
    clients_loaders = [DataLoader(Subset(non_iid_mnist.data, non_iid_mnist.client_data_indices[client]), batch_size=32, shuffle=True)
                       for client in range(num_clients)]

    # 训练模型的示例
    for client_loader in clients_loaders:
        for inputs, labels in client_loader:
            # 训练逻辑在此处实现
            pass

if __name__ == "__main__":
    main()
   