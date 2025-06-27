import torch
import argparse
from graph_mamba import GPSModel

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
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
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



if __name__ == "__main__":
    model = GPSModel(args)
    model = load_checkpoint(model)
