import os

import numpy as np
import torch


def get_preds(model, dataloader, config, pre="train_val"):
    """
    Generate predictions from the model on the provided dataloader and save them.
    This function does not return anything; instead, it aggregates predictions
    and saves them to disk for further analysis or post-processing.
    """
    model.eval()
    all_preds = []
    all_labels = []  # if you want to compare against ground truth

    with torch.no_grad():
        for data in dataloader:
            # Assume the config has a method to send the batch to the correct device
            batch = config.batch_to_device(data, config.DEVICE)
            outputs = model(batch)

            # Extract predictions from outputs. This may need to adapt based on your model.
            if isinstance(outputs, dict):
                preds = outputs.get("predictions", outputs.get("output", None))
            else:
                preds = outputs

            if preds is not None:
                all_preds.append(preds.cpu())

            # Optionally, if your batch includes labels:
            if "labels" in data:
                all_labels.append(data["labels"].cpu())

    # Concatenate all the collected predictions
    if all_preds:
        all_preds = torch.cat(all_preds, dim=0).numpy()
    else:
        all_preds = np.array([])
    if all_labels:
        all_labels = torch.cat(all_labels, dim=0).numpy()
    else:
        all_labels = np.array([])

    # Define the output directory and file name based on config values
    output_dir = os.path.join(config.OUTPUT_DIRECTORY, config.CONFIG, f"fold{config.FOLD}")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{pre}_preds_seed{config.SEED}.npy"), all_preds)

    # Optionally save labels too
    if all_labels.size > 0:
        np.save(os.path.join(output_dir, f"{pre}_labels_seed{config.SEED}.npy"), all_labels)

    # Optionally log summary statistics
    if config.LOCAL_RANK == 0:
        print(f"Saved {pre} predictions with shape: {all_preds.shape}")
