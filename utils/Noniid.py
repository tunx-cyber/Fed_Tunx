import numpy as np
from torch.utils.data import Subset
# every client has a start index and end index for indexing the data 
def dirichlet_setting(num_clients, dataset, alpha=None):
    """
    使用 Dirichlet 分布将数据集分配给客户端

    参数:
    - num_clients: 客户端数量
    - dataset: PyTorch 数据集
    - alpha: Dirichlet 分布的超参数，用于控制数据分配的均匀性

    返回:
    - client_datasets: 每个客户端的数据集列表
    """
    # 获取样本标签
    labels = np.array(dataset.targets)
    num_classes = len(set(labels))
    default_alpha = [0.1] * num_classes
    # 使用 Dirichlet 分布生成客户端的类别比例
    if alpha == None:
        alpha = default_alpha
    dirichlet_samples = np.random.dirichlet(alpha, num_clients)

    # 打印每个客户端的比例
    # for i, proportions in enumerate(dirichlet_samples):
    #     print(f"Client {i + 1} proportions: {proportions}")

    # 根据比例分配样本索引
    allocated_indices = [[] for _ in range(num_clients)]

    # 为每个类别生成索引
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        np.random.shuffle(class_indices)  # 随机打乱索引
        
        # 为每个客户端分配样本
        for client_idx in range(num_clients):
            num_samples = int(dirichlet_samples[client_idx][class_idx] * len(class_indices))
            allocated_indices[client_idx].extend(class_indices[:num_samples])
            class_indices = class_indices[num_samples:]  # 更新剩余的索引

    # 创建客户端数据集
    client_datasets = [Subset(dataset, indices) for indices in allocated_indices]
    return client_datasets

def iid_setting(num_clients, data_size):
    size = data_size//num_clients
    client_dist = []
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + size
        client_dist.append([start_idx, end_idx])
        start_idx = end_idx
    
    return client_dist
