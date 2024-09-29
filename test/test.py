import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# 定义简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载 MNIST 数据集
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # 将数据划分为多个客户端
    client_datasets = [Subset(dataset, range(i * 1200, (i + 1) * 1200)) for i in range(5)]
    return client_datasets

# 训练客户端模型
def train_on_client(model, data_loader, epochs=1):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(epochs):
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

# 联邦平均过程
def fed_avg(global_model, client_models):
    with torch.no_grad():
        for global_param in global_model.parameters():
            global_param.data = torch.mean(
                torch.stack([client_param.data for client_param in client_models]), dim=0
            )

# 主程序
def main():
    client_datasets = load_data()
    global_model = SimpleNN()
    
    # 训练过程
    for epoch in range(5):
        print(f"Global Training Round {epoch + 1}")
        client_models = []
        
        for client_data in client_datasets:
            client_model = SimpleNN()
            client_model.load_state_dict(global_model.state_dict())  # 复制全局模型
            
            client_loader = DataLoader(client_data, batch_size=32, shuffle=True)
            train_on_client(client_model, client_loader, epochs=1)
            client_models.append(client_model)
        
        # 聚合模型参数
        fed_avg(global_model, client_models)

    print("训练完成")

if __name__ == "__main__":
    main()
