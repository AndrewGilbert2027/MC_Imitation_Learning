import torch
from torch.utils.data import DataLoader
from model import ResNetLikeModel
from lazy_loader import LazyLoader
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add CUDA memory management
torch.cuda.empty_cache()  # Clear CUDA cache before starting
if torch.cuda.is_available():
    # Set memory allocation to be more efficient
    torch.backends.cuda.max_split_size_mb = 512
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available GPU memory

# Ensure model save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def train():
    # Initialize dataset and dataloader
    print("Initializing dataset...")
    dataset = LazyLoader(DATA_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Reduce prefetching
    )

    # Initialize model
    print("Initializing model...")
    model = ResNetLikeModel(num_classes=11)  # 11 action classes
    model = model.to(DEVICE)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
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
