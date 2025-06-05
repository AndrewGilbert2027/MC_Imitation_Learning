from collections import deque
import numpy as np


class ImageBuffer:
    def __init__(self, max_size=20):  # Updated max_size to 20
        self.buffer = deque(maxlen=max_size)

    def add_image(self, image):
        """Add an image to the buffer."""
        self.buffer.append(image)

    def get_images(self):
        """Get all images in the buffer."""
        return np.array(self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self):
        """Return the number of images in the buffer."""
        return len(self.buffer)
