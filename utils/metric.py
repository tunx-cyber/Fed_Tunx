'''
随机数种子要一样
'''
'''
test Accuracy:
对于每个client设置model后测试准确率,正确的个数之和除以总测试数据之和。测试数据和训练数据不一样
'''
from utils import client
import numpy as np
def get_accuracy(clients : list[client.Client], global_model):
    correct = []
    total = []
    for client in clients:
        client.model.load_state_dict(global_model.state_dict())
        corr, tol = client.test()
        correct.append(corr)
        total.append(tol)
    return np.sum(correct)*1.0/np.sum(total)


'''
Trainning acc
Trainning loss
'''