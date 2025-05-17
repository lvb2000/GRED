import argparse
import numpy as np
from datasets import load_peptides
from graph_mamba import GPSModel
import torch
from tqdm import tqdm
import logger as log
from sklearn.metrics import average_precision_score

max_nodes = 444
max_hops = 40

parser = argparse.ArgumentParser()
#* model hyper-params
parser.add_argument("--num_layers", default=8, type=int)
parser.add_argument("--num_hops", default=40, type=int)
parser.add_argument("--dim_h", default=88, type=int)
parser.add_argument("--dim_v", default=88, type=int)
parser.add_argument("--r_min", default=0.95, type=float)
parser.add_argument("--r_max", default=1., type=float)
parser.add_argument("--max_phase", default=6.28, type=float)
parser.add_argument("--drop_rate", default=0.2, type=float)
parser.add_argument("--expand", default=1, type=int)
parser.add_argument("--act", default="full-glu", type=str)

#* training hyper-params
parser.add_argument("--lr_min", default=1e-7, type=float)
parser.add_argument("--lr_max", default=1e-3, type=float)
parser.add_argument("--weight_decay", default=0.2, type=float)
parser.add_argument("--lr_factor", default=1., type=float)
parser.add_argument("--name", default="peptides-func", type=str)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--warmup", default=0.05, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()
np.random.seed(args.seed)

if args.num_hops is None:
    args.num_hops = max_hops

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    batch_accumulation = 4
    log.LoggerInit(device,batch_accumulation)
    # train_set[0] = train_xy, train_set[1] = train_distances
    train_set, val_set, test_set = load_peptides(args.name)
    
    model = GPSModel(train_set[0]["x"].shape[-1], args.dim_h,10).to(device)

    train_size = train_set[0]["x"].shape[0]
    train_steps_per_epoch = train_size // args.batch_size
    train_steps_total = train_steps_per_epoch * args.epochs

    val_size = val_set[0]["x"].shape[0]
    val_steps = (val_size - 1) // args.batch_size + 1

    test_size = test_set[0]["x"].shape[0]
    test_steps = (test_size - 1) // args.batch_size + 1

    model.train()
    base_lr = 0.001
    weight_decay = 0.01
    print(f"Optimizer settings:")
    print(f"Learning rate: {base_lr}")
    print(f"Weight decay: {weight_decay}")
    optimizer = torch.optim.AdamW(model.parameters(),lr=base_lr,weight_decay=weight_decay)
    optimizer.zero_grad()

    for e in tqdm(range(args.epochs), desc="Training"):
        model.train()
        # shuffle all training samples (graphs) by indices
        train_indices = np.random.permutation(train_size)
        preds = []
        trues = []
        losses = []
        for s in range(train_steps_per_epoch):
            # go over all training samples with batches
            batch_indices = train_indices[s * args.batch_size:(s + 1) * args.batch_size]
            dist_mask = np.zeros((len(batch_indices), max_hops, max_nodes, max_nodes), dtype=np.bool_)
            for i, idx in enumerate(batch_indices):
                # build up full distance mask for every graph in current batch
                dist_mask[i, :train_set[1][idx].shape[0], :train_set[1][idx].shape[1], :train_set[1][idx].shape[2]] = train_set[1][idx]
            x_batch = torch.from_numpy(train_set[0]["x"][batch_indices]).to(device)
            y_batch = torch.from_numpy(train_set[0]["y"][batch_indices]).to(device)
            node_mask_batch = torch.from_numpy(train_set[0]["node_mask"][batch_indices]).to(device)
            dist_mask_batch = torch.from_numpy(dist_mask[:, :args.num_hops]).to(device)
            # predict
            pred_batch = model(x_batch,node_mask_batch,dist_mask_batch)
            
            loss, pred_score = compute_loss(pred_batch, y_batch)
            loss.backward()

            if ((s + 1) % batch_accumulation == 0) or (s + 1 == train_steps_per_epoch):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            _true = y_batch.detach().cpu().numpy()
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

        val_indices = np.arange(val_size)
        model.eval()
        preds = []
        trues = []
        losses = []
        with torch.no_grad():
            for s in range(val_steps):
                batch_indices = val_indices[s * args.batch_size:(s + 1) * args.batch_size]
                dist_mask = np.zeros((len(batch_indices), max_hops, max_nodes, max_nodes), dtype=np.bool_)
                for i, idx in enumerate(batch_indices):
                    dist_mask[i, :val_set[1][idx].shape[0], :val_set[1][idx].shape[1], :val_set[1][idx].shape[2]] = val_set[1][idx]
                x_batch = torch.from_numpy(val_set[0]["x"][batch_indices]).to(device)
                y_batch = torch.from_numpy(val_set[0]["y"][batch_indices]).to(device)
                node_mask_batch = torch.from_numpy(val_set[0]["node_mask"][batch_indices]).to(device)
                dist_mask_batch = torch.from_numpy(dist_mask[:, :args.num_hops]).to(device)
                
                # predict
                pred_batch = model(x_batch,node_mask_batch,dist_mask_batch)

                loss, pred_score = compute_loss(pred_batch, y_batch)
                _true = y_batch.detach().cpu().numpy()
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