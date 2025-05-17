import wandb

def LoggerInit(device,batch_accumulation):
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
      "batch_accumulation": batch_accumulation
    })

def LoggerUpdate(loss,ap_per_class,ap,epoch,type="train"):
    wandb.log({f"{type}_loss": loss},step=epoch)
    wandb.log({f"{type}_AP_mean": ap},step=epoch)
    wandb.log({f"{type}_AP": {f"Class_{i}": ap_per_class[i] for i in range(len(ap_per_class))}},step=epoch)

def LoggerEnd():
    wandb.finish()