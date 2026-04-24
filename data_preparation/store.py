import wandb
run = wandb.init(
    project="AAI-TASK-1",  # Required
    entity="jaimespencer2-",  # Required
    name="model_save_name",
    group="FFF",
    tags=["stl"],
    config={
        "architecture": [1, 2, 3, 4, 5],
        "transfer_type": [1, 8, 3, 9, 15],
        "learning_rate": [6, 2, 3, 4, 9],
    },
)
