import numpy as np
import pickle
import hashlib
import os.path as osp
import pickle
import shutil

from functools import partial
from scipy.sparse.csgraph import floyd_warshall
from torch_geometric.graphgym.loader import set_dataset_attr

import pandas as pd
import torch
from ogb.utils.mol import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm

def load(name, split):
    data = np.load(f'./data/{name}/{split}.npz')
    dist_mask = np.load(f'./data/{name}/{split}_dist_mask.npy')
    dataset = {
        'x': data['x'],
        'y': data['y'],
        'dist_mask': dist_mask,
        'node_mask': data['node_mask']
    }
    return dataset


def load_sbm(name):
    assert(name in ['PATTERN', 'CLUSTER'])
    train_set = load(name, split='train')
    val_set = load(name, split='val')
    test_set = load(name, split='test')
    return train_set, val_set, test_set

def load_superpixel(name):
    assert(name in ['MNIST', 'CIFAR10'])
    train_set = load(name, split='train')
    val_set = load(name, split='val')
    test_set = load(name, split='test')
    return train_set, val_set, test_set

def load_zinc():
    train_set = load(f'ZINC/subset', split='train')
    train_set['edge_attr'] = np.load(f'./data/ZINC/subset/train_edge_attr.npy')
    val_set = load(f'ZINC/subset', split='val')
    val_set['edge_attr'] = np.load(f'./data/ZINC/subset/val_edge_attr.npy')
    test_set = load(f'ZINC/subset', split='test')
    test_set['edge_attr'] = np.load(f'./data/ZINC/subset/test_edge_attr.npy')
    return train_set, val_set, test_set

class PeptidesFunctionalDataset(InMemoryDataset):
    def __init__(self, root='datasets', smiles2graph=smiles2graph,
                 transform=None, pre_transform=None):
        """
        PyG dataset of 15,535 peptides represented as their molecular graph
        (SMILES) with 10-way multi-task binary classification of their
        functional classes.

        The goal is use the molecular representation of peptides instead
        of amino acid sequence representation ('peptide_seq' field in the file,
        provided for possible baseline benchmarking but not used here) to test
        GNNs' representation capability.

        The 10 classes represent the following functional classes (in order):
            ['antifungal', 'cell_cell_communication', 'anticancer',
            'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
            'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'peptides-functional')

        self.url = 'https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1'
        self.version = '701eb743e899f4d793f0e13c8fa5a1b4'  # MD5 hash of the intended dataset file
        self.url_stratified_split = 'https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1'
        self.md5sum_stratified_split = '5a0114bdadc80b94fc7ae974f13ef061'

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == 'y':
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'peptide_multi_class_dataset.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), 'w').close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir,
                                       'peptide_multi_class_dataset.csv.gz'))
        smiles_list = data_df['smiles']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(
                torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([eval(data_df['labels'].iloc[i])])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root,
                              "splits_random_stratified_peptide.pickle")
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict
    
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

def load_peptides():
    dataset = PeptidesFunctionalDataset('./data')
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    # Estimate directedness based on 10 graphs to save time.
    pre_transform_in_memory(dataset,partial(compute_dist_mask))
    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    setup_standard_split(dataset)
    return dataset

