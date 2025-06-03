import wandb

def LoggerInit(device,args,total_params):
    wandb.init(
      # Set the project where this run will be logged
      project="GRED-Mamba-Tuning",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=args.logging_name,
      # Track hyperparameters and run metadata
      config={
      "architecture": args.architecture,
      "dataset": args.name,
      "epochs": args.epochs,
      "device": device,
      "base_lr": args.base_lr,
      "batch_accumulation": args.batch_accumulation,
      "max_hops": args.max_hops,
      "num_hops":args.num_hops,
      "batch_size": args.batch_size,
      "weight_decay": args.weight_decay,
      "drop_rate": args.drop_rate,
      "warm_up": args.warmup*args.epochs,
      "dim_v": args.dim_v,
      "dim_h": args.dim_h,
      "#parameters": total_params
    })

def LoggerUpdatePeptides(loss,ap_per_class,ap,epoch,type="train"):
    wandb.log({f"{type}_loss": loss},step=epoch)
    wandb.log({f"{type}_AP_mean": ap},step=epoch)
    wandb.log({f"{type}_AP": {f"Class_{i}": ap_per_class[i] for i in range(len(ap_per_class))}},step=epoch)

def LoggerUpdatePictures(loss,accuracy,epoch,type="train"):
    wandb.log({f"{type}_loss": loss},step=epoch)
    wandb.log({f"{type}_accuracy_mean": accuracy},step=epoch)

def LoggerEnd():
    wandb.finish()