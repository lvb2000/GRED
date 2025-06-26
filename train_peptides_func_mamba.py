import argparse
import numpy as np
from datasets.peptides import load_peptides
from datasets.GNNBenchmark import load_GNNBenchmark
from graph_mamba import GPSModel
import torch
from tqdm import tqdm
import logger as log
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader
from torch.optim import Optimizer
from torch import optim
import math
from seed import set_seed

parser = argparse.ArgumentParser()
#* model hyper-params
parser.add_argument("--num_layers", default=8, type=int)
parser.add_argument("--max_hops", default=False,type=bool)
parser.add_argument("--num_hops", default=40, type=int)
parser.add_argument("--dim_h", default=88, type=int)
parser.add_argument("--dim_v", default=88, type=int)
parser.add_argument("--dim_out", default=10, type=int)
parser.add_argument("--drop_rate", default=0, type=float)
parser.add_argument("--expand", default=1, type=int)
parser.add_argument("--act", default="full-glu", type=str)
parser.add_argument("--architecture", default="GRED-MAMBA", type=str)
parser.add_argument("--feature_dimension", default=9, type=int)
parser.add_argument("--pos_enc", default=False, type=bool)
parser.add_argument("--local_model", default=False, type=bool)
parser.add_argument("--checkpoint_dir", type=str)
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

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, last_epoch: int = -1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def main():
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    loaders = create_loader()
    model = GPSModel(args).to(device)
    # Calculate total number of parameters and model size
    total_params = sum(p.numel() for p in model.parameters())
    # Calculate model size in bytes (assuming float32 parameters)
    model_size_bytes = total_params * 4  # 4 bytes per float32
    model_size_mb = model_size_bytes / (1024 * 1024)  # Convert to MB
    args.num_params = total_params
    print(f"Model size: {model_size_mb:.2f} MB ({total_params:,} parameters)")
    log.LoggerInit(device,args,total_params)
    model.train()
    print(f"Optimizer settings:")
    print(f"Learning rate: {args.base_lr}")
    print(f"Weight decay: {args.weight_decay}")
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.base_lr,weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=args.warmup*args.epochs,num_training_steps=args.epochs)
    optimizer.zero_grad()

    if args.max_hops:
        max_graph_diameter = 0
        for iter, batch in enumerate(loaders[0]):
            max_hops = max(batch.k_max)
            if max_hops > max_graph_diameter:
                max_graph_diameter = max_hops
            
        print(f"Max graph diameter in training set: {max_graph_diameter}")

    

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
                # Scale gradients by batch accumulation factor
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data /= args.batch_accumulation
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

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
            log.LoggerUpdatePeptides(mean_loss, ap_per_class, mean_ap, e+1, type="train")
        elif args.name in ['MNIST', 'CIFAR10']:
            # Calculate accuracy
            accuracy = np.mean(preds == trues)
            log.LoggerUpdatePictures(mean_loss, accuracy, e+1, type="train")
            
        # update scheduler
        scheduler.step()

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

        best_score = 0
        if args.name == "peptides-func":
            ap_per_class = average_precision_score(trues, preds, average=None)
            mean_ap = ap_per_class.mean()
            log.LoggerUpdatePeptides(mean_loss, ap_per_class, mean_ap, e+1, type="val")
            if best_score < mean_ap:
                best_score = mean_ap
                log.LoggerLogModel(model,"Average Precision",best_score,e,args.checkpoint_dir)
        elif args.name in ['MNIST', 'CIFAR10']:
            # Calculate accuracy
            accuracy = np.mean(preds == trues)
            log.LoggerUpdatePictures(mean_loss, accuracy, e+1, type="val")
            if best_score < accuracy:
                best_score = accuracy
                log.LoggerLogModel(model,"Accuracy",best_score,e,args.checkpoint_dir)

    log.LoggerEnd()

if __name__ == "__main__":
    main()