import numpy as np
import cv2

# Path to the preprocessed data file
data_file_path = '/home/andrew-gilbert/minecraft_imitation/vpt/data/preprocessed/inputs.npy'

# Load the preprocessed data
try:
    preprocessed_data = np.load(data_file_path, allow_pickle=True)
except FileNotFoundError:
    print(f"Error: File not found at {data_file_path}")
    exit(1)

print(f"Length of preprocessed data: {len(preprocessed_data)}")
print(f"Shape of preprocessed data: {preprocessed_data.shape}")

# Extract the first data point (5-depth input)
if len(preprocessed_data) < 1:
    print("Error: Preprocessed data is empty.")
    exit(1)

first_data_point = preprocessed_data[0]

# Ensure the data point has the expected depth
if len(first_data_point.shape) < 3 or first_data_point.shape[2] < 5:
    print("Error: Data point does not have the expected depth of 5.")
    exit(1)

# Output each channel of the 5-depth input
for i in range(5):
    cv2.imshow(f'Channel {i+1}', first_data_point[:, :, i])
    cv2.waitKey(0)

data_action_file_path = '/home/andrew-gilbert/minecraft_imitation/vpt/data/preprocessed/targets.npy'

# Load the action data
try:
    action_data = np.load(data_action_file_path, allow_pickle=True)
except FileNotFoundError:
    print(f"Error: File not found at {data_action_file_path}")
    exit(1)

print(f"Length of action data: {len(action_data)}")
print(f"Shape of action data: {action_data.shape}")

# Output the first action data point
if len(action_data) < 1:
    print("Error: Action data is empty.")
    exit(1)

first_action_data_point = action_data[0]
print(f"First action data point: {first_action_data_point}")
# Clean up OpenCV windows
cv2.destroyAllWindows()