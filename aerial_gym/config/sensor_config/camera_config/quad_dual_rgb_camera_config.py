from aerial_gym.config.sensor_config.camera_config.base_rgb_camera_config import (
    BaseDepthCameraConfig as BaseRGBCameraConfig,
)
import numpy as np


class QuadDualRGBCameraConfig(BaseRGBCameraConfig):
    num_sensors = 2  # Two cameras: forward-facing and downward-facing

    # Multiple camera positions and orientations
    # Camera 0: Forward-facing camera (rectangular_link position from URDF)
    # Camera 1: Downward-facing camera (downward_camera_link position from URDF)

    nominal_positions = [
        [0.10, 0.0, 0.03],  # Forward camera position
        [0.0, 0.0, -0.06],  # Downward camera position
    ]

    nominal_orientations_euler_deg = [
        [0.0, 0.0, 0.0],     # Forward camera: looking straight ahead
        [0.0, 90.0, 0.0],    # Downward camera: 90° pitch down (perfectly downward)
    ]

    # Disable randomization for multiple cameras (for now)
    randomize_placement = False
