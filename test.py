import gymnasium as gym # Use Gymnasium
import jsbsim_gym.jsbsim_gym # This line makes sure the environment is registered
import imageio as iio
from os import path
from stable_baselines3 import PPO # Assuming you are only visualizing a PPO model)

# Configured for macOS to use Cocoa for rendering (PyGame specific, JSBSimEnv uses its own viewer)

# --- Get the directory of the current script ---
current_script_dir = path.dirname(path.abspath(__file__))
model_path = path.join(current_script_dir, "models", "jsbsim_am_ppo_stacked_lma")
video_mp4_path = path.join(current_script_dir, "video.mp4")
video_gif_path = path.join(current_script_dir, "video.gif")

print(f"Attempting to create environment JSBSim-v0...")

try:
    env = gym.make("JSBSim-v0") # If wrap_jsbsim takes kwargs, pass them here if needed
    print("Environment created successfully.")
except Exception as e:
    print(f"Error creating environment: {e}")
    exit()

print(f"Loading model from: {model_path}")
try:
    model = PPO.load(model_path, env=env)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


print("Starting video recording...")

try:
    with iio.get_writer(video_mp4_path, format="FFMPEG", fps=30, quality=8, codec='libx264') as mp4_writer, \
         iio.get_writer(video_gif_path, format="GIF", fps=10, loop=0) as gif_writer: # GIF fps usually lower

        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        num_steps = 0

        print("Starting episode...")
        for step_count in range(env.spec.max_episode_steps + 50 if env.spec else 1250): # Run for a bit
            num_steps += 1
            try:
                render_data = env.render() # mode='rgb_array' is implied by get_frame in your env
                if render_data is None:
                    print(f"Warning: env.render() returned None at step {step_count}. Skipping frame.")
                else:
                    mp4_writer.append_data(render_data)
                    if step_count % 3 == 0: # Adjust GIF frame skip as needed (e.g., 3 for 10fps from 30fps video)
                        # Downsample for GIF to save space and processing, if desired
                        gif_frame = render_data[::2, ::2, :] # Example: take every 2nd pixel
                        gif_writer.append_data(gif_frame)
            except Exception as e:
                print(f"Error during rendering or appending frame at step {step_count}: {e}")
                # break

            action, _states = model.predict(obs, deterministic=True)
            
            # Environment step now returns 5 values
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step_count % 50 == 0 or terminated or truncated:
                print(f"Step: {num_steps}, Reward: {reward:.3f}, Total Reward: {total_reward:.3f}, Terminated: {terminated}, Truncated: {truncated}")

            if terminated or truncated:
                print(f"Episode finished after {num_steps} steps.")
                print(f"Final reward: {total_reward}")
                if terminated: print("Reason: Terminated (e.g., crash or goal)")
                if truncated: print("Reason: Truncated (e.g., time limit)")
                break
        
        if not (terminated or truncated):
            print(f"Episode reached max test steps ({num_steps}) without termination/truncation.")

except Exception as e:
    print(f"An error occurred during the episode loop: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("Closing environment.")
    env.close()
    print(f"MP4 video saved to: {video_mp4_path}")
    print(f"GIF video saved to: {video_gif_path}")

print("Testing script finished.")