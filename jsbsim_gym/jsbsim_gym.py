import jsbsim
import gym # Using OpenAI Gym
import numpy as np
import collections 

from .visualization.rendering import Viewer, load_mesh, load_shader, RenderObject, Grid
from .visualization.quaternion import Quaternion

STATE_FORMAT = [
    "position/lat-gc-rad", "position/long-gc-rad", "position/h-sl-meters",
    "velocities/mach", "aero/alpha-rad", "aero/beta-rad",
    "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec",
    "attitude/phi-rad", "attitude/theta-rad", "attitude/psi-rad",
]

EPSILON = 1e-5
SINGLE_OBS_LOW = np.array([
    -np.inf, -np.inf, -np.inf,       # lat, lon, alt
    0,                               # mach
    -np.pi - EPSILON, -np.pi - EPSILON, # alpha, beta
    -np.inf, -np.inf, -np.inf,       # p, q, r
    -np.pi - EPSILON,                # phi (roll)
    -np.pi/2 - EPSILON,              # theta (pitch)
    -np.pi - EPSILON,                # psi (yaw)
    -np.inf, -np.inf, 0,             # goal_x, goal_y, goal_z
], dtype=np.float32)

SINGLE_OBS_HIGH = np.array([
    np.inf, np.inf, np.inf,          # lat, lon, alt
    np.inf,                          # mach
    np.pi + EPSILON, np.pi + EPSILON,    # alpha, beta
    np.inf, np.inf, np.inf,          # p, q, r
    np.pi + EPSILON,                 # phi (roll)
    np.pi/2 + EPSILON,               # theta (pitch)
    np.pi + EPSILON,                 # psi (yaw)
    np.inf, np.inf, np.inf,          # goal_x, goal_y, goal_z
], dtype=np.float32)

RADIUS = 6.3781e6
NUM_STACKED_FRAMES = 10

def normalize_angle_mpi_pi(angle_rad):
    """Normalizes an angle in radians to the range [-pi, pi)."""
    if np.isnan(angle_rad) or np.isinf(angle_rad): # Handle non-finite inputs
        # Decide on a handling strategy: return 0, raise error, or return NaN
        # For observation space, returning a fixed valid value or NaN (if space allows)
        # print(f"Warning: normalize_angle_mpi_pi received non-finite value: {angle_rad}")
        return 0.0 # Or np.nan if your space is [-inf, inf] for angles, but it's not.
                   # Returning 0.0 is a simple fix to avoid NaN propagation if not desired.

    angle_rad = angle_rad % (2 * np.pi)
    if angle_rad >= np.pi:
        angle_rad -= 2 * np.pi
    # After x % (2*pi), if x was negative, angle_rad could be in (-2*pi, 0].
    # E.g. -0.1 % (2*pi) might give something like 6.18...
    # Then 6.18... - 2*pi gives -0.1. This part is fine.
    # The check `elif angle_rad < -np.pi:` was originally commented out, it's generally not needed
    # if the first modulo and the `if angle_rad >= np.pi:` are correct.
    return angle_rad

def normalize_angle_0_2pi(angle_rad):
    """Normalizes an angle in radians to the range [0, 2*pi)."""
    return angle_rad % (2 * np.pi)

class JSBSimEnv(gym.Env):
    def __init__(self, root='.'):
        super().__init__()
        self.num_stacked_frames = NUM_STACKED_FRAMES
        single_obs_dim = len(SINGLE_OBS_LOW)
        self.observation_space = gym.spaces.Box(
            low=np.tile(SINGLE_OBS_LOW, (self.num_stacked_frames, 1)),
            high=np.tile(SINGLE_OBS_HIGH, (self.num_stacked_frames, 1)),
            shape=(self.num_stacked_frames, single_obs_dim),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(np.array([-1,-1,-1,0], dtype=np.float32), 1, (4,), dtype=np.float32)
        self.obs_buffer = collections.deque(maxlen=self.num_stacked_frames)
        self.simulation = jsbsim.FGFDMExec(root, None)
        self.simulation.set_debug_level(0)
        self.simulation.load_model('f16')
        self._set_initial_conditions()
        self.simulation.run_ic()
        self.down_sample = 4
        self.current_step = 0 # To potentially implement truncation for max_episode_steps
        self.max_episode_steps = 1200 # Matching gym.register, can be used for manual truncation
        self.state = np.zeros(12) 
        self.goal = np.zeros(3)
        self.dg = 100
        self.viewer = None

    def _set_initial_conditions(self):
        self.simulation.set_property_value('propulsion/set-running', -1)
        self.simulation.set_property_value('ic/u-fps', 900.)
        self.simulation.set_property_value('ic/h-sl-ft', 5000)
    
    def _get_current_single_observation(self):
        # print(f"\n--- Step {self.current_step} _get_current_single_observation ---") # For grouping prints
        for i, property_name in enumerate(STATE_FORMAT):
            raw_value = self.simulation.get_property_value(property_name)
            # if property_name in ["attitude/phi-rad", "attitude/theta-rad", "attitude/psi-rad"]:
            #     print(f"Raw {property_name}: {raw_value}")
            self.state[i] = raw_value

        # Store raw angles before normalization for comparison
        # raw_phi = self.state[9]
        # raw_theta = self.state[10]
        # raw_psi = self.state[11]

        # Normalize angles
        self.state[9] = normalize_angle_mpi_pi(self.state[9])   # Roll
        self.state[10] = normalize_angle_mpi_pi(self.state[10])  # Pitch
        self.state[11] = normalize_angle_mpi_pi(self.state[11])  # Yaw

        # Print comparison IF a value seems problematic from your diagnostic
        # Example: if your diagnostic said psi was 6.28...
        # if abs(raw_psi - 6.283) < 0.01 or abs(self.state[11] - 6.283) < 0.01 : # If raw or normalized psi is near 2*pi
        #     print(f"DEBUG ANGLE (PSI): Step {self.current_step} | Raw Psi: {raw_psi:.6f} -> Normalized Psi: {self.state[11]:.6f}")
        # if abs(raw_phi - 6.283) < 0.01 or abs(self.state[9] - 6.283) < 0.01 :
        #     print(f"DEBUG ANGLE (PHI): Step {self.current_step} | Raw Phi: {raw_phi:.6f} -> Normalized Phi: {self.state[9]:.6f}")


        self.state[:2] *= RADIUS
        final_obs_single_frame = np.hstack([self.state, self.goal]).astype(np.float32)
        
        # Print the problematic angle from the final observation frame if it's still wrong
        # if abs(final_obs_single_frame[11] - 6.283) < 0.01 : # Check psi in final frame
        #     print(f"FINAL FRAME CHECK (PSI): Step {self.current_step} | Psi in final_obs_single_frame[11]: {final_obs_single_frame[11]:.6f}")
        # if abs(final_obs_single_frame[9] - 6.283) < 0.01 : # Check phi in final frame
        #     print(f"FINAL FRAME CHECK (PHI): Step {self.current_step} | Phi in final_obs_single_frame[9]: {final_obs_single_frame[9]:.6f}")
        return final_obs_single_frame

    def step(self, action):
        self.current_step += 1
        roll_cmd, pitch_cmd, yaw_cmd, throttle = action
        self.simulation.set_property_value("fcs/aileron-cmd-norm", roll_cmd)
        self.simulation.set_property_value("fcs/elevator-cmd-norm", pitch_cmd)
        self.simulation.set_property_value("fcs/rudder-cmd-norm", yaw_cmd)
        self.simulation.set_property_value("fcs/throttle-cmd-norm", throttle)
        for _ in range(self.down_sample):
            self.simulation.set_property_value("propulsion/tank/contents-lbs", 1000)
            self.simulation.set_property_value("propulsion/tank[1]/contents-lbs", 1000)
            self.simulation.set_property_value("gear/gear-cmd-norm", 0.0)
            self.simulation.set_property_value("gear/gear-pos-norm", 0.0)
            self.simulation.run()
        current_single_obs = self._get_current_single_observation()
        self.obs_buffer.append(current_single_obs)
        
        reward = 0
        terminated = False # Gymnasium: task is over (success or fail)
        truncated = False  # Gymnasium: episode ended due to external factor (e.g. time limit)

        latest_state_part = current_single_obs[:12]
        if latest_state_part[2] < 10: # Crash
            reward = -10
            terminated = True

        if np.sqrt(np.sum((latest_state_part[:2] - self.goal[:2])**2)) < self.dg and \
           abs(latest_state_part[2] - self.goal[2]) < self.dg: # Reached goal
            reward = 10
            terminated = True
        
        # Handle truncation due to max episode steps, if not already terminated
        if not terminated and self.current_step >= self.max_episode_steps:
            truncated = True
            # Optional: Add a small penalty or specific handling for truncation if needed
            # reward -= 1 # Example if you want to penalize time-outs differently

        obs = np.array(self.obs_buffer, dtype=np.float32)
        info = {}
        
        # Check if the observation is within the defined space
        if not self.observation_space.contains(obs):
            bad = np.logical_or(obs < self.observation_space.low,
                                obs > self.observation_space.high) & \
                np.isfinite(obs)              # ignore inf where allowed
            idx = np.where(bad)
            print("Out‑of‑bounds indices:", idx, "values:", obs[idx])
            raise ValueError("Observation outside space")

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.current_step = 0 # Reset step counter
        self.simulation.run_ic()
        self.simulation.set_property_value('propulsion/set-running', -1)
        
        rng = np.random.default_rng(seed)
        distance = rng.random() * 9000 + 1000
        bearing = rng.random() * 2 * np.pi
        altitude = rng.random() * 3000
        self.goal[:2] = np.cos(bearing), np.sin(bearing)
        self.goal[:2] *= distance
        self.goal[2] = altitude
        initial_single_obs = self._get_current_single_observation()
        self.obs_buffer.clear() 
        for _ in range(self.num_stacked_frames):
            self.obs_buffer.append(np.copy(initial_single_obs))
        
        return np.array(self.obs_buffer, dtype=np.float32), {}
    
    def render(self, mode='human'):
        scale = 1e-3

        if self.viewer is None:
            self.viewer = Viewer(1280, 720)

            f16_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
            self.f16 = RenderObject(f16_mesh)
            self.f16.transform.scale = 1/30
            self.f16.color = 0, 0, .4

            goal_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "cylinder.obj")
            self.cylinder = RenderObject(goal_mesh)
            self.cylinder.transform.scale = scale * 100
            self.cylinder.color = 0, .4, 0

            self.viewer.objects.append(self.f16)
            self.viewer.objects.append(self.cylinder)
            self.viewer.objects.append(Grid(self.viewer.ctx, self.viewer.unlit, 21, 1.))
        
        # Rough conversion from lat/long to meters
        x, y, z = self.state[:3] * scale

        self.f16.transform.z = x 
        self.f16.transform.x = -y
        self.f16.transform.y = z

        rot = Quaternion.from_euler(*self.state[9:])
        rot = Quaternion(rot.w, -rot.y, -rot.z, rot.x)
        self.f16.transform.rotation = rot

        # self.viewer.set_view(-y , z + 1, x - 3, Quaternion.from_euler(np.pi/12, 0, 0, mode=1))

        x, y, z = self.goal * scale

        self.cylinder.transform.z = x
        self.cylinder.transform.x = -y
        self.cylinder.transform.y = z

        r = self.f16.transform.position - self.cylinder.transform.position
        rhat = r/np.linalg.norm(r)
        x,y,z = r
        yaw = np.arctan2(-x,-z)
        pitch = np.arctan2(-y, np.sqrt(x**2 + z**2))


        self.viewer.set_view(*(r + self.cylinder.transform.position + rhat + np.array([0, .33, 0])), Quaternion.from_euler(-pitch, yaw, 0, mode=1))
        self.viewer.render()

        if mode == 'rgb_array':
            return self.viewer.get_frame()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class PositionReward(gym.Wrapper):
    def __init__(self, env, gain):
        super().__init__(env)
        self.gain = gain
    
    def step(self, action):
        # The wrapped env (JSBSimEnv) now returns 5 values
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_single_obs = obs[-1] 
        displacement = current_single_obs[-3:] - current_single_obs[:3]
        distance = np.linalg.norm(displacement)
        
        # Add reward shaping. Only add if not yet terminated or truncated by the base env.
        # Or, you might want to add it regardless, depending on desired behavior.
        # Let's assume for now we add it regardless, as it's a shaping reward.
        reward += self.gain * (self.last_distance - distance)
        self.last_distance = distance
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs) 
        current_single_obs = obs[-1]
        displacement = current_single_obs[-3:] - current_single_obs[:3]
        self.last_distance = np.linalg.norm(displacement)
        return obs, info

def wrap_jsbsim(**kwargs):
    # The JSBSimEnv is now Gymnasium-like in its step/reset returns
    # The PositionReward wrapper also maintains this.
    # SB3's shimmy will still wrap it if it detects gym.Env, but the API calls should align better.
    return PositionReward(JSBSimEnv(**kwargs), 1e-2)

gym.register(
    id="JSBSim-v0",
    entry_point=wrap_jsbsim,
    max_episode_steps=1200 # This is used by TimeLimit wrapper
)

if __name__ == "__main__":
    from time import sleep
    env = gym.make("JSBSim-v0") 
    
    # Test with gymnasium API if possible, or let SB3 handle it
    # Since we're using gym.make, SB3 will wrap it.
    obs, info = env.reset(seed=42) 
    print("Initial stacked observation shape:", obs.shape)
    env.render()
    
    for i in range(env.max_episode_steps + 50): # Run a bit longer to test truncation
        action = env.action_space.sample() 
        
        # When using the raw env (even if wrapped by SB3's gym->gymnasium compat layer),
        # the return will be 5 values if the underlying env provides it.
        # SB3's VecEnv will then convert terminated/truncated into a single `dones` flag for its buffer.
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated # Combine for local loop logic
        except ValueError: # Fallback if a wrapper still expects 4 (less likely now)
            obs, reward, done, info = env.step(action)
            print("Warning: Env step returned 4 values, expected 5. Check wrapper chain.")


        if i % 10 == 0:
            print(f"Step {i}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        env.render()
        
        if done:
            print("Episode finished after {} timesteps".format(i+1))
            if terminated:
                print("Reason: Terminated (e.g. goal reached or crash)")
            if truncated:
                print("Reason: Truncated (e.g. time limit)")
            obs, info = env.reset(seed=i) 
            print("Reset. New stacked observation shape:", obs.shape)
        sleep(1/30)
    env.close()