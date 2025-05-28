import os
import pickle
import numpy as np
from torch_geometric.datasets import LRGBDataset
from functools import partial
from multiprocessing import Pool
from scipy.sparse.csgraph import floyd_warshall
max_num_nodes = 444
func_sp = partial(floyd_warshall, directed=False, unweighted=True)

def gen_dist_mask(adj):
    dist = func_sp(adj)
    dist = np.where(np.isfinite(dist), dist, -1).astype(np.int32)
    k_max = dist.max() + 1
    dist_mask = np.stack([(dist == k) for k in range(k_max)])
    return dist_mask

def main():
    for name in ['peptides-func', 'peptides-struct']:
        for split in ['train', 'val', 'test']:
            dataset = LRGBDataset(root='./data', name=name, split=split)
            dataset_as_dict = {key: [] for key in ['x', 'y', 'adj']}
            for g in dataset:
                num_nodes = g['x'].shape[0]
                adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                edge_index = g['edge_index'].numpy()
                adj[edge_index[0], edge_index[1]] = 1.
                dataset_as_dict['adj'].append(adj)
                x = g['x'].numpy().astype(np.int32)
                dataset_as_dict['x'].append(x)
                dataset_as_dict['y'].append(g['y'].numpy())
            
            adjs = dataset_as_dict.pop('adj')
            with Pool(25) as p: #! adjust according to your machine
                dist_mask = p.map(gen_dist_mask, adjs)
            
            pickle.dump(dist_mask, open(f'./data/{name}/{split}_dist_mask.pkl', 'wb'))
            np.savez(f'./data/{name}/{split}.npz',
                    x = np.array(dataset_as_dict['x'], dtype=object),
                    y = np.concatenate(dataset_as_dict['y']),
                    allow_pickle=True)

if __name__ == '__main__':
    if not os.path.exists('./data'):
        os.mkdir('./data')
    main()