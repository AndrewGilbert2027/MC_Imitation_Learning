import torch
import gym
import minerl
import numpy as np
from model import ResNetLikeModel
from image_buffer import ImageBuffer
import cv2


def resize_image(image, target_size=(256, 256)):
    """
    Resize an image to the target size.

    Args:
        image (np.ndarray): Input image array.
        target_size (tuple): Desired output size (width, height).

    Returns:
        np.ndarray: Resized image.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def preprocess_image(image):
    """
    Preprocess an image by resizing it to 256x256 pixels and normalizing RGB values.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        np.ndarray: Preprocessed image.
    """
    resized_image = resize_image(image, target_size=(256, 256))  # Resize to 256x256
    normalized_image = resized_image / 255.0  # Normalize pixel values to [0, 1]
    return normalized_image


def load_model(model_path, num_classes, device):
    """
    Loads a PyTorch model from the specified path.

    Args:
        model_path (str): Path to the saved model file.
        num_classes (int): Number of output classes for the model.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: Loaded model.
    """
    # Initialize the model architecture
    model = ResNetLikeModel(num_classes=num_classes)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load just the model state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Move the model to the specified device
    model.to(device)
    print("Model loaded successfully.")
    return model

if __name__ == "__main__":
    # Path to the saved model
    model_path = "models/model_epoch_9.pth"
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    num_classes = 11  # Adjusted for binary camera directions
    model = load_model(model_path, num_classes, device)

    # Initialize the image buffer
    image_buffer = ImageBuffer(max_size=20)  # Updated max_size to 20

    # Initialize the environment
    env = gym.make('MineRLBasaltFindCave-v0')
    env.seed(2143)

    obs = env.reset()
    # Add the initial observation to the image buffer
    image_buffer.add_image(preprocess_image(obs['pov']))
    print(f"shape of image: {image_buffer.get_images().shape}")
    done = False
    env.render()


    # Main loop
    while not done:
        count = 0
        if (len(image_buffer.get_images()) < 20 and count < 40):  # Updated depth check to 20
            # If there are not enough images in the buffer, skip the step
            obs, reward, done, _ = env.step({'noop': [], 'ESC': 0, 'forward': 1, 'right': 1})
            image_buffer.add_image(preprocess_image(obs['pov']))
            print(image_buffer.get_images().shape)
            env.render()
            count += 1
            continue
        else:
            # Get the images from the buffer
            images = image_buffer.get_images()
            # Convert to tensor and permute dimensions to match model input (unsqueeze to add batch dimension)
            input_tensor = torch.tensor(images, dtype=torch.float32)  # (20, 256, 256, 3)
            input_tensor = input_tensor.permute(0, 3, 1, 2)           # (20, 3, 256, 256)
            input_tensor = input_tensor.reshape(1, 60, 256, 256).to(device)  # (1, 60, 256, 256)
            
            # Predict using the model
            output = model.predict(input_tensor)

            # Apply sigmoid to binary values (indices 0-6)
            binary_probs = torch.sigmoid(output[:, :7])  # Extract probabilities for binary actions
            binary_actions = torch.bernoulli(binary_probs).cpu().detach().numpy().astype(int)  # Sample binary actions

            # Map the binary output to actions
            action = {
                'noop': [],
                'ESC': 0,  # Default action to avoid ending the episode
                'forward': binary_actions[0][0],
                'back': binary_actions[0][1],
                'left': binary_actions[0][2],
                'right': binary_actions[0][3],
                'jump': binary_actions[0][4],
                'sprint': binary_actions[0][5],
                'attack': binary_actions[0][6],
                'camera': [0.0, 0.0],
            }

            # Camera movement based on probabilities
            camera_probs = torch.sigmoid(output[:, 7:]).cpu().detach().numpy()
            if camera_probs[0][0] > np.random.rand():
                action['camera'][0] = -0.5  # Look up
            if camera_probs[0][1] > np.random.rand():
                action['camera'][0] = 0.5  # Look down
            if camera_probs[0][2] > np.random.rand():
                action['camera'][1] = -0.5  # Look left
            if camera_probs[0][3] > np.random.rand():
                action['camera'][1] = 0.5  # Look right

            obs, reward, done, _ = env.step(action)
            image_buffer.add_image(preprocess_image(obs['pov']))
            env.render()
            count += 1
    
    # Example: Print model summary
    print(model)
