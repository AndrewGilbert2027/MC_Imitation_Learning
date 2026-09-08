import torch
from torch.utils.data import DataLoader, Subset
from model import ResNetLikeModel
from lazy_loader import LazyLoader
from split import positive_weights, split_indices_by_video
import torch.nn as nn
import os
from tqdm import tqdm

# Configuration
DATA_DIR = "./vpt/data/labeller-training"
MODEL_SAVE_DIR = "./models"
BATCH_SIZE = 16  # Reduced batch size
NUM_WORKERS = 4   # Reduced number of workers
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VAL_FRACTION = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add CUDA memory management
torch.cuda.empty_cache()  # Clear CUDA cache before starting
if torch.cuda.is_available():
    # Set memory allocation to be more efficient
    torch.backends.cuda.max_split_size_mb = 512
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available GPU memory

# Ensure model save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def loader_for(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Reduce prefetching
    )


def training_labels(dataset, indices):
    """
    Read the action labels for the training split only.

    load_action touches the JSON and not the video, so this is cheap even
    though the dataset itself is lazy. Reading the training split alone keeps
    validation statistics out of the class weights.
    """
    return torch.stack([
        dataset.load_action(dataset.sample_pairs[i][1], dataset.sample_pairs[i][2])
        for i in indices
    ])


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    """
    Validation loss and accuracy, next to the score for predicting "no action"
    on every control.

    With sparse controls most frames are mostly zeros, so raw accuracy looks
    high for a model that has learned nothing. The baseline is what makes the
    number readable.
    """
    model.eval()
    total_loss, correct, predicted_positive, total = 0.0, 0, 0, 0
    always_negative_correct = 0

    for frames, actions in dataloader:
        batch_size, channels, time_steps, height, width = frames.shape
        frames = frames.reshape(batch_size, channels * time_steps, height, width)
        frames, actions = frames.to(DEVICE), actions.to(DEVICE)

        outputs = model(frames)
        total_loss += criterion(outputs, actions).item()

        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == actions).sum().item()
        predicted_positive += predictions.sum().item()
        always_negative_correct += (actions == 0).sum().item()
        total += actions.numel()

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": correct / max(total, 1),
        "baseline_accuracy": always_negative_correct / max(total, 1),
        "predicted_positive_rate": predicted_positive / max(total, 1),
    }


def train():
    # Initialize dataset and dataloader
    print("Initializing dataset...")
    dataset = LazyLoader(DATA_DIR)

    # Whole videos are held out rather than random frames: neighbouring frames
    # of one recording are near-identical, so a random split would score the
    # model on near-copies of what it trained on.
    train_indices, val_indices = split_indices_by_video(
        dataset.sample_pairs, val_fraction=VAL_FRACTION, seed=SEED
    )
    print(f"Train samples: {len(train_indices)} | Validation samples: {len(val_indices)}")

    dataloader = loader_for(dataset, train_indices, shuffle=True)
    val_dataloader = loader_for(dataset, val_indices, shuffle=False) if val_indices else None

    # Initialize model
    print("Initializing model...")
    model = ResNetLikeModel(num_classes=11)  # 11 action classes
    model = model.to(DEVICE)

    # Most frames hold most controls at zero, so an unweighted loss is minimised
    # by predicting nothing. Weighting the positive class by how rare it is
    # keeps the sparse controls in the gradient.
    print("Computing class weights...")
    pos_weight = positive_weights(training_labels(dataset, train_indices)).to(DEVICE)
    print("pos_weight:", [round(w, 2) for w in pos_weight.tolist()])

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    val_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch_idx, (frames, actions) in enumerate(progress_bar):
            # frames shape from loader: (batch_size, C, T, H, W)
            # Need to reshape to: (batch_size, C*T, H, W)
            batch_size, channels, time_steps, height, width = frames.shape
            frames = frames.reshape(batch_size, channels * time_steps, height, width)
            
            # Move data to device
            frames = frames.to(DEVICE)
            actions = actions.to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, actions)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        # Epoch statistics
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Average Loss: {avg_loss:.4f}")

        if val_dataloader is not None:
            stats = evaluate(model, val_dataloader, val_criterion)
            print(
                f"    val loss {stats['loss']:.4f} | "
                f"accuracy {stats['accuracy']:.4f} "
                f"(always-negative baseline {stats['baseline_accuracy']:.4f}) | "
                f"predicted-positive rate {stats['predicted_positive_rate']:.4f}"
            )

        # Save checkpoint
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)

if __name__ == "__main__":
    train()
