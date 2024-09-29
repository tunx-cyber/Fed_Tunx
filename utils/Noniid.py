import numpy as np

# every client has a start index and end index for indexing the data 
def dirichlet_setting(num_clients : int, data_size : int, _alpha = None):
    alpha = np.ones(num_clients)
    if(_alpha != None):
        alpha = _alpha
    dist = np.random.dirichlet(alpha=alpha, size=1).flatten()

    client_data_sizes = (dist * data_size).astype(int)

    client_dist = []
    
    start_idx = 0
    for size in client_data_sizes:
        end_idx = start_idx + size
        client_dist.append([start_idx, end_idx])
        start_idx = end_idx
    
    return client_dist

def iid_setting(num_clients, data_size):
    size = data_size//num_clients
    client_dist = []
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + size
        client_dist.append([start_idx, end_idx])
        start_idx = end_idx
    
    return client_dist
