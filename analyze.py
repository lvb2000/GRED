import torch
import argparse
from graph_mamba_first_layer import GPSModel
from train_peptides_func_mamba import compute_loss, create_loader
import numpy as np
from sklearn.metrics import average_precision_score
from einops import einsum, rearrange

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



def load_checkpoint(model):
    """
    Loads a torch model from a file in the 'Checkpoints' folder.

    Args:
        model_class: The class of the model to instantiate.
        filename (str): The name of the checkpoint file.

    Returns:
        model: The loaded model with weights restored.
    """
    checkpoint_path = input("Enter the path to the checkpoint file: ")
    #checkpoint_path="Checkpoints/best_model_epoch_132.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    #for idx, (k, v) in enumerate(checkpoint.items()):
    #    print(f"Index: {idx}, Key: {k}")
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for idx, (k, v) in enumerate(checkpoint.items()) if idx <= 29}
    model_state_dict.update(filtered_state_dict)
    #model.load_state_dict(checkpoint)
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    print("Model loaded with partial weights successfully!")
    return model

def print_model_structure(model):
    """
    Prints the model structure, including all submodules and their parameters.

    Args:
        model: The PyTorch model to inspect.
    """
    print("Model Structure:")
    print(model)
    print("\nNamed Modules:")
    for name, module in model.named_modules():
        print(f"Module: {name} -> {module.__class__.__name__}")
    print("\nNamed Parameters:")
    for name, param in model.named_parameters():
        print(f"Parameter: {name} | Shape: {tuple(param.shape)}")

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


def analyze_B(dt,B,u):
    seqlen = 40
    print(f"Shape of B: {B.shape}")
    l2_norms_per_token_per_sample = torch.linalg.norm(B, dim=1)
    print(f"Shape after calculating L2 norm for each token: {l2_norms_per_token_per_sample.shape}")
    average_l2_norm_over_batch = torch.mean(l2_norms_per_token_per_sample, dim=0)
    print(f"Final shape (average L2 norm per sequence position): {average_l2_norm_over_batch.shape}")
    print(f"Average L2 Norm values over sequence: \n{average_l2_norm_over_batch}")
    dt = rearrange(dt, "b d l -> b l d", l=seqlen)
    B = rearrange(B, "b dstate l -> b l dstate", l=seqlen).contiguous()
    u = rearrange(u, "b d l -> b l d", l=seqlen)
    deltaB_u = einsum(dt, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
    print(f"Shape of deltaB: {deltaB_u.shape}")
    l2_norms_per_token_per_sample = torch.linalg.norm(deltaB_u, dim=3)
    average_l2_norm_over_batch = torch.mean(l2_norms_per_token_per_sample, dim=2)
    average_l2_norm_over_batch = torch.mean(average_l2_norm_over_batch, dim=0)
    print(f"Final shape (average L2 norm per sequence position): {average_l2_norm_over_batch.shape}")
    print(f"Average L2 Norm values over sequence: \n{average_l2_norm_over_batch}")

def test_model_matrix(model,loader,device):
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
            dt,A,B,C,u = model(batch,dist_mask,device)
            analyze_B(dt,B,u)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPSModel(args)
    model = load_checkpoint(model)
    print_model_structure(model)
    loaders = create_loader()
    test_model_matrix(model.to(device),loaders[2],device)
