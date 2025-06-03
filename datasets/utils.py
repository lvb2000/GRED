import numpy as np
from functools import partial
from scipy.sparse.csgraph import floyd_warshall
from tqdm import tqdm
from torch_geometric.graphgym.loader import set_dataset_attr

def set_dataset_splits(dataset, splits):
    """Set given splits to the dataset object.

    Args:
        dataset: PyG dataset object
        splits: List of train/val/test split indices

    Raises:
        ValueError: If any pair of splits has intersecting indices
    """
    # First check whether splits intersect and raise error if so.
    for i in range(len(splits) - 1):
        for j in range(i + 1, len(splits)):
            n_intersect = len(set(splits[i]) & set(splits[j]))
            if n_intersect != 0:
                raise ValueError(
                    f"Splits must not have intersecting indices: "
                    f"split #{i} (n = {len(splits[i])}) and "
                    f"split #{j} (n = {len(splits[j])}) have "
                    f"{n_intersect} intersecting indices"
                )
    # split on graph level
    split_names = [
        'train_graph_index', 'val_graph_index', 'test_graph_index'
    ]
    for split_name, split_index in zip(split_names, splits):
        set_dataset_attr(dataset, split_name, split_index, len(split_index))

def setup_standard_split(dataset):
    """Select a standard split.

    Use standard splits that come with the dataset. Pick one split based on the
    ``split_index`` from the config file if multiple splits are available.

    GNNBenchmarkDatasets have splits that are not prespecified as masks. Therefore,
    they are handled differently and are first processed to generate the masks.

    Raises:
        ValueError: If any one of train/val/test mask is missing.
        IndexError: If the ``split_index`` is greater or equal to the total
            number of splits available.
    """

    for split_name in 'train_graph_index', 'val_graph_index', 'test_graph_index':
        if not hasattr(dataset.data, split_name):
            raise ValueError(f"Missing '{split_name}' for standard split")

def pre_transform_in_memory(dataset, transform_func):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i)) 
                for i in tqdm(range(len(dataset)), 
                desc="Pre-transforming")]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)

func_sp = partial(floyd_warshall, directed=False, unweighted=True)

def gen_dist_mask(adj):
    dist = func_sp(adj)
    dist = np.where(np.isfinite(dist), dist, -1).astype(np.int32)
    k_max = dist.max() + 1
    dist_mask = np.stack([(dist == k) for k in range(k_max)])
    return dist_mask,k_max
        
def compute_dist_mask(g):
    num_nodes = g.x.shape[0]
    g.graph_nodes = num_nodes
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[g.edge_index[0], g.edge_index[1]] = 1.
    g.dist_mask,g.k_max = gen_dist_mask(adj)
    return g
