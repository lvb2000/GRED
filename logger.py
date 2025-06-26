import wandb
import torch
import os

def LoggerInit(device,args,total_params):
    wandb.init(
      # Set the project where this run will be logged
      project="GRED-Mamba-Results",
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
      "#parameters": total_params,
      "pos_enc": args.pos_enc,
      "local_model": args.local_model,
      "seed": args.seed,
      "checkpoint_dir": args.checkpoint_dir
    })

def LoggerLogModel(model, metric_name, metric_score, epoch, checkpoint_dir):
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_save_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_save_path)
    artifact = wandb.Artifact(
        name=f"model-checkpoint",
        type="model",
        description=f"Best model based on {metric_name} at epoch {epoch+1}",
        metadata={"epoch": epoch + 1, metric_name: metric_score}
    )
    artifact.add_file(model_save_path)
    # Use an alias to easily refer to the "best" model
    # This will override the "best" alias if a new best model is found
    wandb.log_artifact(artifact, aliases=["latest", "best_model"])
    

def LoggerUpdatePeptides(loss,ap_per_class,ap,epoch,type="train"):
    wandb.log({f"{type}_loss": loss},step=epoch)
    wandb.log({f"{type}_AP_mean": ap},step=epoch)
    wandb.log({f"{type}_AP": {f"Class_{i}": ap_per_class[i] for i in range(len(ap_per_class))}},step=epoch)

def LoggerUpdatePictures(loss,accuracy,epoch,type="train"):
    wandb.log({f"{type}_loss": loss},step=epoch)
    wandb.log({f"{type}_accuracy_mean": accuracy},step=epoch)

def LoggerEnd():
    wandb.finish()