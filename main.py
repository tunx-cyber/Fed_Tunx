'''
For test
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets
# 简单的线性模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 1)  # 假设输入为2维，输出为1维

    def forward(self, x):
        return self.fc(x)

# 模拟客户端的本地模型更新
def local_update(model, data_size):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 模拟一些本地数据
    for _ in range(5):  # 模拟5个训练轮次
        inputs = torch.randn(data_size, 2)  # 随机输入
        targets = torch.randn(data_size, 1)  # 随机目标值
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    return model.state_dict()

# 聚合函数
def federated_averaging(client_updates, client_sizes):
    global_weights = {}
    total_weight = sum(client_sizes)

    for key in client_updates[0].keys():
        global_weights[key] = sum(client_updates[i][key] * client_sizes[i] for i in range(len(client_updates))) / total_weight

    return global_weights

# 模拟客户端数量和数据样本数量
num_clients = 3
client_data_sizes = [50, 75, 25]  # 各客户端的数据样本数量
client_updates = []

# 每个客户端进行本地更新
for i in range(num_clients):
    model = SimpleModel()  # 每个客户端都有一个新模型
    local_weights = local_update(model, client_data_sizes[i])
    client_updates.append(local_weights)

print(client_updates[0].items())
# 进行聚合
global_weights = federated_averaging(client_updates, client_data_sizes)

# 输出结果
print("聚合后的全局权重:")
for key, value in global_weights.items():
    print(f"{key}: {value}")

# 添加测试函数
def test_model(model, test_dataset):
    model.eval()  # 设置模型为评估模式
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# 创建测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# 进行 Federated Learning
for round in range(5):  # 训练轮数
    local_states = []
    for client_data in clients_data:
        local_state = train_on_client(client_data, model)
        local_states.append(local_state)

    # FedAvg: 计算全局模型
    global_state_dict = federated_average(local_states)
    model.load_state_dict(global_state_dict)

    # 测试模型准确率
    accuracy = test_model(model, test_dataset)
    print(f"Round {round + 1}, Accuracy: {accuracy:.2f}%")

print("训练完成！")
