import numpy as np
import torch
import time
from torch_geometric.nn import knn_graph
# from torch_geometric.nn import knn
from torch_cluster import knn
from typing import Optional

def my_knn_graph(x: torch.Tensor, y:torch.Tensor, k: int, 
                batch_x: Optional[torch.Tensor] = None,
                batch_y: Optional[torch.Tensor] = None,
                loop: bool = False, flow: str = 'source_to_target',
                cosine: bool = False, num_workers: int = 1) -> torch.Tensor:

    assert flow in ['source_to_target', 'target_to_source']
    # Finds for each element in :obj:`y` the :obj:`k` nearest points in obj:`x`.
    edge_index = knn(x, y, k if loop else k + 1, batch_x, batch_y, cosine,
                     num_workers)

    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)


# 10 points, each point has two dimensions
x = torch.randint(0, 100, (50000, 500))
batch = torch.tensor([0 for _ in range(x.shape[0])])
print (x)

# use knn_graph
st = time.time()
edge_index = knn_graph(x, batch=batch, k=2, cosine=False, loop=False)
print (edge_index.shape)
print("edge_index: ", edge_index)

print("Finish calculate edge index, the shape is {}, time cost: {:4f}".format(edge_index.shape, time.time()-st))


# use divide
n=int(x.shape[0]/2+ 1)
x_l_indices = np.arange(n)
x_u_indices = np.array(list(set(np.arange(x.shape[0])) - set(x_l_indices)))
x_l = x[x_l_indices]
x_u = x[x_u_indices]

st = time.time()
batch = torch.tensor([0 for _ in range(x_l.shape[0])])
edge_index_t = my_knn_graph(x_l, x_l, batch_x=batch, batch_y=batch, k=2)
print ("l-2-l edge index: ", edge_index_t)
#edge_index = torch.zeros_like(edge_index_t)
new_edge_index_l0 = [x_l_indices[i] for i in list(edge_index_t[0])]
new_edge_index_l1 = [x_l_indices[i] for i in list(edge_index_t[1])]
l2l_edge_index = torch.tensor([new_edge_index_l0, new_edge_index_l1])
print ("replaced edge index: ", l2l_edge_index)

batch_x = torch.tensor([0 for _ in range(x_l.shape[0])])
batch_y = torch.tensor([0 for _ in range(x_u.shape[0])])
edge_index_t = my_knn_graph(x_l, x_u, batch_x=batch_x, batch_y=batch_y, k=2)
print ("u-2-l edge index: ", edge_index_t)
new_edge_index_l0 = [x_l_indices[i] for i in list(edge_index_t[0])]
new_edge_index_l1 = [x_u_indices[i] for i in list(edge_index_t[1])]
u2l_edge_index = torch.tensor([new_edge_index_l0, new_edge_index_l1])
print ("replaced edge index: ", u2l_edge_index)

edge_index = torch.cat([l2l_edge_index, u2l_edge_index], dim=1)
print ("final edge_index, ", u2l_edge_index)
print("Finish calculate edge index, the shape is {}, time cost: {:4f}".format(edge_index.shape, time.time()-st))