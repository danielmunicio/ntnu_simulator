from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
from aerial_gym.vla_planner.llava_planner import LlavaPlanner

if __name__ == "__main__":
    args = get_args()
    logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="simple_env",
        # robot_name="base_quadrotor_with_rgb",
        # robot_name='base_quadrotor_with_stereo_camera',
        robot_name='quad_camera',
        controller_name="lee_velocity_control",
        args=None,
        device="cuda:0",
        num_envs=1,
        headless=False,
        use_warp=False,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    # actions[:, 0] = 5.0
    # actions[:, 1] = -5.0
    # actions[:, 3] = np.pi
    # actions[:, 2] = 0.0
    env_manager.reset()
    depth_frames_forward = []
    depth_frames_downward = []
    planner = LlavaPlanner(f'you are the controller for a drone, here is an image, respond in one word with a direction of either left, right, or straight,'
    'that I should go too, to get to the Red Cube. Your current velocity is ')

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title('Dual Camera View')

    for i in range(2000):
        env_manager.step(actions=actions)
        env_manager.render(render_components="sensors")

        # Debug prints every 50 steps
        if i == 0:
            logger.info(f"Available keys: {env_manager.global_tensor_dict.keys()}")
        if i % 50 == 0:
            # Access robot state tensors
            position = env_manager.global_tensor_dict["robot_position"][0]
            velocity = env_manager.global_tensor_dict["robot_body_linvel"][0]
            ang_vel = env_manager.global_tensor_dict["robot_body_angvel"][0]
            logger.info(f"\n--- Step {i} ---")
            logger.info(f"Position (xyz): {position.cpu().numpy()}")
            logger.info(f"Velocity (xyz): {velocity.cpu().numpy()}")
            logger.info(f"Angular vel: {ang_vel.cpu().numpy()}")
            logger.info(f"Actions: {actions[0].cpu().numpy()}")
        # print("ENV MANAGER: ",env_manager.global_tensor_dict.keys())

        try:
            # Get RGB images from both cameras
            # Camera 0: Forward-facing
            image_forward = (
                env_manager.global_tensor_dict["rgb_pixels"][0, 0].cpu().numpy()
            ).astype(np.uint8)

            # Camera 1: Downward-facing
            image_downward = (
                env_manager.global_tensor_dict["rgb_pixels"][0, 1].cpu().numpy()
            ).astype(np.uint8)

            # Drop alpha channel for both cameras
            image_forward_rgb = image_forward[:, :, :3]
            image_downward_rgb = image_downward[:, :, :3]

            # Clear and update both subplots
            ax1.clear()
            ax2.clear()

            ax1.imshow(image_forward_rgb)
            ax1.set_title('Forward Camera')
            ax1.axis('off')

            ax2.imshow(image_downward_rgb)
            ax2.set_title('Downward Camera')
            ax2.axis('off')

            plt.tight_layout()
            plt.savefig('current_frame_dual.png')

            if i % 100 == 0:
                # direction = planner.get_direction(image_forward_rgb)
                direction = 'bonk'
                if direction == 'Forward' or direction == 'Straight':
                    actions[:, 0] += 1
                    actions[:, 1] = 0
                elif direction == 'Right':
                    actions[:, 1]  = -1
                    actions[:, 0] = 0
                elif direction == 'Left':
                    actions[:, 1] += 1
                    actions[:, 0] = 0

            # for now just overwrite to get drone to test stable hover
            actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
            plt.pause(0.001)  # Non-blocking update
        except Exception as e:
            logger.error("Error in getting images")
            logger.error("Seems like the image tensors have not been created yet.")
            logger.error("This is likely due to absence of a functional camera in the environment")
            raise e

        depth_image_forward = Image.fromarray(image_forward_rgb)
        depth_image_downward = Image.fromarray(image_downward_rgb)
        depth_frames_forward.append(depth_image_forward)
        depth_frames_downward.append(depth_image_downward)

    # Save GIFs for both cameras
    depth_frames_forward[0].save(
        "rgb_forward_gif.gif",
        save_all=True,
        append_images=depth_frames_forward[1:],
        duration=100,
        loop=0,
    )
    depth_frames_downward[0].save(
        "rgb_downward_gif.gif",
        save_all=True,
        append_images=depth_frames_downward[1:],
        duration=100,
        loop=0,
    )
    depth_frames_forward[0].save(f"forward_frame_{i}.png")
    depth_frames_downward[0].save(f"downward_frame_{i}.png")

