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
    depth_frames = []
    planner = LlavaPlanner('here is an image, respond in one word with a direction of either left, right, or straight,'
    'that I should go too, to get to the Red Cube')

    for i in range(2000):
        env_manager.step(actions=actions)
        env_manager.render(render_components="sensors")
        # print("ENV MANAGER: ",env_manager.global_tensor_dict.keys())

        try:
            # Get RGB image from tensor (shape: H x W x 4 for RGBA/BGRA)
            image1 = (
                env_manager.global_tensor_dict["rgb_pixels"][0, 0].cpu().numpy()
            ).astype(np.uint8)

            # Drop alpha channel, then reverse BGR to RGB
            image1_rgb = image1[:, :, :3]  # Drop alpha channel
            # image1_rgb = image1_rgb[:, :, ::-1]  # Reverse channel order (BGR->RGB)
            fig = plt.figure(num='Front Camera View')
            plt.clf()  # Clear previous plot
            plt.imshow(image1_rgb)
            plt.savefig('current_frame.png')

            if i % 100 == 0:
                direction = planner.get_direction(image1_rgb)
                if direction == 'Forward' or direction == 'Straight':
                    actions[:, 0] += 1
                    actions[:, 1] = 0
                elif direction == 'Right':
                    actions[:, 1]  = -1
                    actions[:, 0] = 0
                elif direction == 'Left':
                    actions[:, 1] += 1
                    actions[:, 0] = 0


            plt.pause(0.001)  # Non-blocking update
        except Exception as e:
            logger.error("Error in getting images")
            logger.error("Seems like the image tensors have not been created yet.")
            logger.error("This is likely due to absence of a functional camera in the environment")
            raise e

        depth_image1 = Image.fromarray(image1_rgb)
        depth_frames.append(depth_image1)

    depth_frames[0].save(
        "rgb_gif.gif",
        save_all=True,
        append_images=depth_frames[1:],
        duration=100,
        loop=0,
    )
    depth_frames[0].save(f"depth_frame_{i}.png")

