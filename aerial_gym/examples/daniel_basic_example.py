from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args
from PIL import Image 
import numpy as np

if __name__ == "__main__":
    args = get_args()
    logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="simple_env",
        # robot_name="base_quadrotor_with_rgb",
        robot_name='base_quadrotor_with_stereo_camera',
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        num_envs=2,
        headless=False,
        use_warp=False,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    # actions[:, 2] = 1
    env_manager.reset()
    depth_frames = []
    for i in range(1000):
        env_manager.step(actions=actions)
        env_manager.render(render_components="sensors")
        # print("ENV MANAGER: ",env_manager.global_tensor_dict.keys())

        try:
            image1 = (
                255.0 * env_manager.global_tensor_dict["rgb_pixels"][0, 0].cpu().numpy()
            ).astype(np.uint8)
        except Exception as e:
            logger.error("Error in getting images")
            logger.error("Seems like the image tensors have not been created yet.")
            logger.error("This is likely due to absence of a functional camera in the environment")
            raise e

        depth_image1 = Image.fromarray((image1))
        depth_frames.append(depth_image1)

    depth_frames[0].save(
        "rgb_gif.gif",
        save_all=True,
        append_images=depth_frames[1:],
        duration=100,
        loop=0,
    )
    depth_frames[0].save(f"depth_frame_{i}.png")
