import numpy as np
import cv2
from collections import deque

class FrameProcessor:
    def __init__(self, frame_height=240, frame_width=256, stack_size=4, grayscale=True, resize=False, resize_shape=(84, 84)):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.grayscale = grayscale
        self.resize = resize
        self.resize_shape = resize_shape
        self.frame_height = frame_height
        self.frame_width = frame_width

    def preprocess(self, frame):
        """Preprocess a single frame: grayscale, resize, normalize."""
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert RGB to Grayscale
            frame = np.expand_dims(frame, axis=-1)           # (H, W, 1)

        if self.resize:
            frame = cv2.resize(frame, self.resize_shape, interpolation=cv2.INTER_AREA)  # Resize frame

        frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return frame

    def reset(self, frame):
        """Reset the frame stack with an initial frame."""
        processed_frame = self.preprocess(frame)
        self.frames = deque([processed_frame] * self.stack_size, maxlen=self.stack_size)
        return self.get_stacked_frames()

    def step(self, frame):
        """Add a new frame to the stack."""
        processed_frame = self.preprocess(frame)
        self.frames.append(processed_frame)
        return self.get_stacked_frames()

    def get_stacked_frames(self):
        """Return stacked frames as (channels, height, width) for CNN."""
        stacked = np.concatenate(list(self.frames), axis=-1)  # (H, W, stacked_channels)
        stacked = stacked.transpose(2, 0, 1)  # (stacked_channels, H, W)
        return stacked
