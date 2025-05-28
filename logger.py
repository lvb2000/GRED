import wandb

def LoggerInit(device,args):
    wandb.init(
      # Set the project where this run will be logged
      project="GRED-Mamba",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name="baseline",
      # Track hyperparameters and run metadata
      config={
      "architecture": "GRED-Mamba",
      "dataset": "Peptides-functional",
      "epochs": 200,
      "device": device,
      "base_lr": args.base_lr,
      "batch_accumulation": args.batch_accumulation,
      "max_hops": args.max_hops,
      "num_hops":args.num_hops,
      "batch_size": args.batch_size,
      "weight_decay": args.weight_decay
    })

def LoggerUpdate(loss,ap_per_class,ap,epoch,type="train"):
    wandb.log({f"{type}_loss": loss},step=epoch)
    wandb.log({f"{type}_AP_mean": ap},step=epoch)
    wandb.log({f"{type}_AP": {f"Class_{i}": ap_per_class[i] for i in range(len(ap_per_class))}},step=epoch)

def LoggerEnd():
    wandb.finish()