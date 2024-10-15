import matplotlib.pyplot as plt

def draw_clients_acc_bar(num_clients, accs):
    x = range(num_clients)
    plt.bar(x, accs, color='skyblue')
    plt.title("local test accuracy")
    plt.xlabel("client id")
    plt.ylabel("accuracy")
    plt.show()

def draw_trainning_acc(rounds, accs, title):
    plt.plot(rounds, accs, label='test_acc', color='blue')
    plt.title(title)
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.show()

#TODO
def draw_clients_data_dist(clients_data_dist):
    pass