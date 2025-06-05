# Minecraft Imitation Learning

This project implements imitation learning for Minecraft gameplay using PyTorch. It records gameplay, processes the data, and trains a model to imitate player actions. This repository is based off of this research paper by OpenAI: https://openai.com/index/vpt/

## Project Structure
```
minecraft_imitation/
├── vpt/
│   ├── main.py              # Recording script for gameplay data
│   ├── lazy_dataset.py      # Dataset loader for training
│   ├── data_preprocessor.py # Preprocesses recorded data (not used anymore but could be helpful)
│   └── data/
│       └── labeller-training/ # Recorded gameplay data
├── models/                  # Saved model checkpoints
├── train.py                # Main training script
├── model.py                # Model architecture definition
└── load_model.py           # Model inference script
```

## Requirements
- Python 3.10+
- PyTorch
- OpenCV
- MineRL
- NumPy
- tqdm

## Tips
I would personally recommend using pyenv for this project as it allows easy switching and solves a lot of compatibility issues. The python version that I use for this project was 3.10.14

## Usage

### 1. Recording Gameplay Data
To record gameplay data, navigate to the `vpt` directory and run:
```bash
cd vpt
python main.py
```
This will:
- Launch Minecraft with MineRL
- Record gameplay frames and actions
- Save data to `vpt/data/labeller-training/`

### 2. Training the Model
To train the model on recorded data, from the root directory run:
```bash
python train.py
```
This will:
- Load the recorded data
- Train the ResNet-like model
- Save checkpoints to `models/`

### 3. Running Inference
To use a trained model for inference:
```bash
python load_model.py
```

## Data Format
- **Video Files**: MP4 format, 256x256 resolution
- **Action Files**: JSON format with the following features:
  - Movement: forward, back, left, right
  - Actions: sprint, jump, attack
  - Camera: look up, down, left, right

## Model Architecture
The model uses a ResNet-like architecture with:
- Input: 20 frames (60 channels - RGB x 20 frames)
- 5 residual blocks with increasing channels
- Output: 11 binary actions

## Training Configuration
- Batch Size: 16
- Learning Rate: 0.001
- Optimizer: Adam
- Loss: Binary Cross-Entropy with Logits
- GPU Memory Usage: 80% of available memory

## Notes
- Ensure sufficient disk space for recording 
- GPU recommended for training
- Adjust batch size and workers based on available memory