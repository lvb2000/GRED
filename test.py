import torch
import argparse
from graph_mamba import GPSModel
import numpy as np
from sklearn.metrics import average_precision_score
from einops import einsum, rearrange
from datasets.peptides import load_peptides
from torch_geometric.loader import DataLoader
from datasets.GNNBenchmark import load_GNNBenchmark
import wandb

parser = argparse.ArgumentParser()
#* model hyper-params
parser.add_argument("--num_layers", default=8, type=int)
parser.add_argument("--max_hops", default=False,type=bool)
parser.add_argument("--num_hops", default=40, type=int)
parser.add_argument("--dim_h", default=88, type=int)
parser.add_argument("--dim_v", default=16, type=int)
parser.add_argument("--dim_out", default=10, type=int)
parser.add_argument("--drop_rate", default=0, type=float)
parser.add_argument("--expand", default=1, type=int)
parser.add_argument("--act", default="full-glu", type=str)
parser.add_argument("--architecture", default="GRED-MAMBA", type=str)
parser.add_argument("--feature_dimension", default=9, type=int)
parser.add_argument("--pos_enc", default=False, type=bool)
parser.add_argument("--local_model", default=False, type=bool)
parser.add_argument("--checkpoint_dir",default="", type=str)
parser.add_argument("--model_file", default="", type=str)
#* training hyper-params
parser.add_argument("--batch_accumulation", default=2, type=int)
parser.add_argument("--base_lr", default=0.001, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--name", default="peptides-func", type=str)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--warmup", default=0.05, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--logging_name", default="baseline", type=str)
args = parser.parse_args()



def load_checkpoint(model):
    """
    Loads a torch model from a file in the 'Checkpoints' folder.

    Args:
        model_class: The class of the model to instantiate.
        filename (str): The name of the checkpoint file.

    Returns:
        model: The loaded model with weights restored.
    """
    run = wandb.run
    artifact = run.use_artifact(args.checkpoint_dir, type='model')
    artifact_dir = artifact.download()
    model_path = f"{artifact_dir}/{args.model_file}"
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print("Model loaded successfully!")
    return model

def compute_loss(pred, true):
    """
    Compute loss and prediction score

    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Grou

    Returns: Loss, normalized prediction score

    """
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    # CrossEntropy Loss
    # multiclass
    if pred.ndim > 1 and true.ndim == 1:
        pred = torch.nn.functional.log_softmax(pred, dim=-1)
        return torch.nn.functional.nll_loss(pred, true), pred
    # binary or multilabel
    else:
        true = true.float()
        return bce_loss(pred, true), torch.sigmoid(pred)


def test_model(model,loader,device):
    model.eval()
    preds = []
    trues = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            # Calculate the max hops in the current batch
            max_hops = max(batch.k_max)
            # Calculate the largest number of nodes in the current batch
            max_nodes = max(batch.graph_nodes)
            batch_size = len(batch.dist_mask)
            dist_mask = np.zeros((batch_size, max_hops, max_nodes, max_nodes), dtype=np.bool_)
            for idx in range(batch_size):
                # build up full distance mask for every graph in current batch
                dist_mask[idx, :batch.dist_mask[idx].shape[0], :batch.dist_mask[idx].shape[1], :batch.dist_mask[idx].shape[2]] = batch.dist_mask[idx]
            if not args.max_hops:
                dist_mask= dist_mask[:, :args.num_hops]
            dist_mask = torch.from_numpy(dist_mask).to(device)
            batch.to(device)
            
            # predict
            pred_batch = model(batch,dist_mask,device)

            loss, pred_score = compute_loss(pred_batch, batch.y)
            _true = batch.y.detach().cpu().numpy()
            _pred = pred_score.detach().cpu().numpy()
            _loss = loss.detach().cpu().numpy()
            if args.name in ['MNIST', 'CIFAR10']:
                _pred = np.argmax(_pred, axis=1)
            trues.append(_true)
            preds.append(_pred)
            losses.append(_loss)

    preds = np.vstack(preds)
    trues = np.vstack(trues)
    # Remove rows where preds or trues contain NaNs
    if args.name == "peptides-func":
        mask = ~(np.isnan(preds).any(axis=1) | np.isnan(trues).any(axis=1))
        preds = preds[mask]
        trues = trues[mask]

    losses = np.array(losses)
    mean_loss = losses.mean()

    if args.name == "peptides-func":
        ap_per_class = average_precision_score(trues, preds, average=None)
        mean_ap = ap_per_class.mean()
        print(f"Mean loss: {mean_loss:.4f}")
        print(f"Average Precision per class: {ap_per_class}")
        print(f"Mean Average Precision: {mean_ap:.4f}")
    elif args.name in ['MNIST', 'CIFAR10']:
        # Calculate accuracy
        accuracy = np.mean(preds == trues)
        print(f"Mean loss: {mean_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

def get_loader(dataset, batch_size, shuffle=True):
    pw = False
    loader_train = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=0,
                                  pin_memory=True, persistent_workers=pw, drop_last=True)

    return loader_train

def create_loader():
    """Create data loader object.

    Returns: List of PyTorch data loaders

    """
    dataset = None
    if args.name == "peptides-func":
        dataset = load_peptides(args.pos_enc)
    if args.name == 'MNIST' or  args.name == 'CIFAR10':
        dataset = load_GNNBenchmark(args.name)
    if dataset is None:
        print("Error: Dataset could not be loaded. Please check the dataset name and configuration.")
        exit(1)
    # train loader
    id = dataset.data['train_graph_index']
    loaders = [
        get_loader(dataset[id], args.batch_size,
                    shuffle=True)
    ]
    delattr(dataset.data, 'train_graph_index')

    # val and test loaders
    for i in range(2):
        split_names = ['val_graph_index', 'test_graph_index']
        id = dataset.data[split_names[i]]
        loaders.append(
            get_loader(dataset[id], args.batch_size,
                        shuffle=False))
        delattr(dataset.data, split_names[i])

    return loaders



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init()
    model = GPSModel(args)
    model = load_checkpoint(model)
    # print_model_structure(model)
    loaders = create_loader()
    test_model(model.to(device),loaders[2],device)
    wandb.finish()