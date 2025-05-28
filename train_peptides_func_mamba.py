import argparse
import numpy as np
from datasets import load_peptides
from graph_mamba import GPSModel
import torch
from tqdm import tqdm
import logger as log
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser()
#* model hyper-params
parser.add_argument("--num_layers", default=8, type=int)
parser.add_argument("--max_hops", default=False,type=bool)
parser.add_argument("--num_hops", default=100, type=int)
parser.add_argument("--dim_h", default=88, type=int)
parser.add_argument("--dim_v", default=88, type=int)
parser.add_argument("--r_min", default=0.95, type=float)
parser.add_argument("--r_max", default=1., type=float)
parser.add_argument("--max_phase", default=6.28, type=float)
parser.add_argument("--drop_rate", default=0.2, type=float)
parser.add_argument("--expand", default=1, type=int)
parser.add_argument("--act", default="full-glu", type=str)

#* training hyper-params
parser.add_argument("--batch_accumulation", default=10, type=int)
parser.add_argument("--base_lr", default=0.001, type=float)
parser.add_argument("--lr_min", default=1e-7, type=float)
parser.add_argument("--lr_max", default=1e-3, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--lr_factor", default=1., type=float)
parser.add_argument("--name", default="peptides-func", type=str)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--warmup", default=0.05, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()
np.random.seed(args.seed)


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
        pred = torch.nn.functional.F.log_softmax(pred, dim=-1)
        return torch.nn.functional.F.nll_loss(pred, true), pred
    # binary or multilabel
    else:
        true = true.float()
        return bce_loss(pred, true), torch.sigmoid(pred)


def get_loader(dataset, batch_size, shuffle=True):
    pw = False
    loader_train = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=4,
                                  pin_memory=True, persistent_workers=pw)

    return loader_train


def create_loader():
    """Create data loader object.

    Returns: List of PyTorch data loaders

    """
    dataset = load_peptides()
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    log.LoggerInit(device,args)
    loaders = create_loader()
    model = GPSModel(9, args.dim_h,10).to(device)

    model.train()
    base_lr = args.base_lr
    weight_decay = args.weight_decay
    print(f"Optimizer settings:")
    print(f"Learning rate: {base_lr}")
    print(f"Weight decay: {weight_decay}")
    optimizer = torch.optim.AdamW(model.parameters(),lr=base_lr,weight_decay=weight_decay)
    optimizer.zero_grad()

    for e in tqdm(range(args.epochs), desc="Training"):
        model.train()
        preds = []
        trues = []
        losses = []
        for iter, batch in enumerate(loaders[0]):
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
            loss.backward()

            if ((iter + 1) % args.batch_accumulation == 0) or (iter + 1 == len(loaders[0])):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            _true = batch.y.detach().cpu().numpy()
            _pred = pred_score.detach().cpu().numpy()
            _loss = loss.detach().cpu().numpy()
            trues.append(_true)
            preds.append(_pred)
            losses.append(_loss)

        preds = np.vstack(preds)
        trues = np.vstack(trues)
        # Remove rows where preds or trues contain NaNs
        mask = ~(np.isnan(preds).any(axis=1) | np.isnan(trues).any(axis=1))
        preds = preds[mask]
        trues = trues[mask]
        ap_per_class = average_precision_score(trues, preds, average=None)
        mean_ap = ap_per_class.mean()
        losses = np.array(losses)
        mean_loss = losses.mean()
        log.LoggerUpdate(mean_loss,ap_per_class, mean_ap,e+1,type="train")

        model.eval()
        preds = []
        trues = []
        losses = []
        with torch.no_grad():
            for batch in loaders[1]:
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
                trues.append(_true)
                preds.append(_pred)
                losses.append(_loss)

        preds = np.vstack(preds)
        trues = np.vstack(trues)
        # Remove rows where preds or trues contain NaNs
        mask = ~(np.isnan(preds).any(axis=1) | np.isnan(trues).any(axis=1))
        preds = preds[mask]
        trues = trues[mask]
        ap_per_class = average_precision_score(trues, preds, average=None)
        mean_ap = ap_per_class.mean()
        losses = np.array(losses)
        mean_loss = losses.mean()
        log.LoggerUpdate(mean_loss,ap_per_class, mean_ap,e+1,type="val")

    log.LoggerEnd()

if __name__ == "__main__":
    main()