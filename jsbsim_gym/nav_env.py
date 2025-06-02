import jsbsim
import gymnasium as gym # Changed from OpenAI Gym to Gymnasium for modern API
import numpy as np
import collections

# Assuming visualization modules are in a 'visualization' subdirectory relative to this file.
from .visualization.rendering import Viewer, load_mesh, load_shader, RenderObject, Grid
from .visualization.quaternion import Quaternion

# Defines the JSBSim properties used to construct a single frame of the environment's state.
# These properties are extracted directly from the JSBSim simulation.
STATE_FORMAT = [
    "position/lat-gc-rad",      # Latitude in geocentric radians
    "position/long-gc-rad",     # Longitude in geocentric radians
    "position/h-sl-meters",     # Altitude above sea level in meters
    "velocities/mach",          # Mach number
    "aero/alpha-rad",           # Angle of attack in radians
    "aero/beta-rad",            # Sideslip angle in radians
    "velocities/p-rad_sec",     # Roll rate in radians per second
    "velocities/q-rad_sec",     # Pitch rate in radians per second
    "velocities/r-rad_sec",     # Yaw rate in radians per second
    "attitude/phi-rad",         # Roll angle in radians
    "attitude/theta-rad",       # Pitch angle in radians
    "attitude/psi-rad",         # Yaw angle in radians
]

# Small epsilon value for floating point comparisons in observation space bounds.
EPSILON = 1e-5

# Defines the lower bounds for a single observation frame.
# This includes the 12 state variables from STATE_FORMAT plus 3 goal coordinates (x, y, z).
SINGLE_OBS_LOW = np.array([
    -np.inf, -np.inf, -np.inf,              # Latitude, Longitude (scaled), Altitude
    0,                                      # Mach number (non-negative)
    -np.pi - EPSILON, -np.pi - EPSILON,     # Alpha, Beta (typically within [-pi, pi])
    -np.inf, -np.inf, -np.inf,              # Roll, Pitch, Yaw rates (unbounded)
    -np.pi - EPSILON,                       # Roll angle (phi)
    -np.pi/2 - EPSILON,                     # Pitch angle (theta, typically [-pi/2, pi/2])
    -np.pi - EPSILON,                       # Yaw angle (psi)
    -np.inf, -np.inf, 0,                    # Goal X, Y (meters, unbounded), Goal Z (altitude, non-negative)
], dtype=np.float32)

# Defines the upper bounds for a single observation frame.
SINGLE_OBS_HIGH = np.array([
    np.inf, np.inf, np.inf,                 # Latitude, Longitude (scaled), Altitude
    np.inf,                                 # Mach number
    np.pi + EPSILON, np.pi + EPSILON,       # Alpha, Beta
    np.inf, np.inf, np.inf,                 # Roll, Pitch, Yaw rates
    np.pi + EPSILON,                        # Roll angle (phi)
    np.pi/2 + EPSILON,                      # Pitch angle (theta)
    np.pi + EPSILON,                        # Yaw angle (psi)
    np.inf, np.inf, np.inf,                 # Goal X, Y, Z (meters)
], dtype=np.float32)

# Mean Earth radius in meters, used for converting lat/lon to approximate Cartesian coordinates.
RADIUS = 6.3781e6
# Number of consecutive observation frames stacked to form the agent's input.
NUM_STACKED_FRAMES = 10

def normalize_angle_mpi_pi(angle_rad: float) -> float:
    """
    Normalizes an angle in radians to the range [-pi, pi).

    Args:
        angle_rad: The angle in radians.

    Returns:
        The normalized angle in radians within the range [-pi, pi).
        Returns 0.0 for non-finite (NaN or Inf) inputs to prevent errors.
    """
    if np.isnan(angle_rad) or np.isinf(angle_rad):
        # Non-finite inputs can cause issues in modulo operations or subsequent calculations.
        # Returning 0.0 is a safe default, assuming it's an acceptable neutral value.
        return 0.0
    angle_rad = angle_rad % (2 * np.pi)
    if angle_rad >= np.pi:
        angle_rad -= 2 * np.pi
    return angle_rad

def normalize_angle_0_2pi(angle_rad: float) -> float:
    """
    Normalizes an angle in radians to the range [0, 2*pi).

    Args:
        angle_rad: The angle in radians.

    Returns:
        The normalized angle in radians within the range [0, 2*pi).
        Returns input % (2 * np.pi), which handles non-finite inputs by
        returning NaN if input is NaN, or potentially non-finite if input is Inf.
        Consider adding explicit non-finite handling if necessary.
    """
    return angle_rad % (2 * np.pi)

class JSBSimEnv(gym.Env):
    """
    A Gymnasium environment for flight simulation using JSBSim with an F-16 model.

    The environment involves navigating the F-16 aircraft to a randomly generated
    goal position. The observation space is a stack of recent simulation states,
    and the action space consists of normalized flight control inputs.

    Attributes:
        observation_space (gym.spaces.Box): The observation space, consisting of
            `NUM_STACKED_FRAMES` frames, each with 15 features (12 aircraft state
            variables + 3 goal coordinates).
        action_space (gym.spaces.Box): The action space, representing normalized
            commands for aileron, elevator, rudder, and throttle.
        num_stacked_frames (int): Number of frames stacked for observations.
        simulation (jsbsim.FGFDMExec): The JSBSim flight dynamics model executive.
        current_step (int): The current step count within the episode.
        max_episode_steps (int): The maximum number of steps per episode.
        state (np.ndarray): Internal buffer for the 12 aircraft state variables.
        goal (np.ndarray): Current 3D goal coordinates (x, y, z in meters).
        dg (float): Tolerance distance (radius of goal cylinder) in meters for reaching the goal.
        viewer (Viewer): Custom renderer instance for visualization.
        obs_buffer (collections.deque): Buffer to store recent observation frames.
        down_sample (int): Number of JSBSim simulation steps per environment step.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30} # Common metadata

    def __init__(self, root: str = '.'):
        """
        Initializes the JSBSim flight environment.

        Args:
            root (str): The root directory for JSBSim, used to find aircraft models
                        and other simulation data. Defaults to the current directory.
        """
        super().__init__()
        self.num_stacked_frames = NUM_STACKED_FRAMES
        single_obs_dim = len(SINGLE_OBS_LOW) # Should be 15 (12 state + 3 goal)

        # Define observation space: a stack of frames, each frame has single_obs_dim features.
        self.observation_space = gym.spaces.Box(
            low=np.tile(SINGLE_OBS_LOW, (self.num_stacked_frames, 1)),
            high=np.tile(SINGLE_OBS_HIGH, (self.num_stacked_frames, 1)),
            shape=(self.num_stacked_frames, single_obs_dim),
            dtype=np.float32
        )
        # Define action space: [roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd]
        # Commands are normalized: roll/pitch/yaw in [-1, 1], throttle in [0, 1].
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32), # Corrected high bound for throttle
            shape=(4,),
            dtype=np.float32
        )

        self.obs_buffer = collections.deque(maxlen=self.num_stacked_frames)
        self.simulation = jsbsim.FGFDMExec(root, None)
        self.simulation.set_debug_level(0) # Disable verbose JSBSim console output.
        self.simulation.load_model('f16') # Load the F-16 aircraft model.
        self._set_initial_conditions()
        self.simulation.run_ic() # Apply initial conditions.

        self.down_sample = 4 # JSBSim steps per environment step.
        self.current_step = 0
        self.max_episode_steps = 1200 # Maximum steps before truncation.

        self.state = np.zeros(12, dtype=np.float32) # Aircraft state vector.
        self.goal = np.zeros(3, dtype=np.float32)  # Goal position [x, y, z] in meters.
        self.dg = 100.0  # Goal achievement radius in meters.
        self.viewer = None # Renderer instance, initialized on first render call.

    def _set_initial_conditions(self):
        """Sets the initial flight conditions for the simulation."""
        self.simulation.set_property_value('propulsion/set-running', -1) # Start engines.
        self.simulation.set_property_value('ic/u-fps', 900.0) # Initial forward speed (ft/s).
        self.simulation.set_property_value('ic/h-sl-ft', 5000.0) # Initial altitude (ft).

    def _get_current_single_observation(self) -> np.ndarray:
        """
        Retrieves the current aircraft state from JSBSim, normalizes angles,
        scales positions, and combines it with the goal position to form a single
        observation frame.

        Returns:
            np.ndarray: A single observation frame (15 features) as a float32 NumPy array.
        """
        for i, property_name in enumerate(STATE_FORMAT):
            self.state[i] = self.simulation.get_property_value(property_name)

        # Normalize Euler angles (phi, theta, psi) to ensure consistency.
        # Indices 9, 10, 11 in self.state correspond to phi, theta, psi.
        self.state[9] = normalize_angle_mpi_pi(self.state[9])   # Roll (phi)
        self.state[10] = normalize_angle_mpi_pi(self.state[10])  # Pitch (theta)
        self.state[11] = normalize_angle_mpi_pi(self.state[11])  # Yaw (psi)

        # Convert latitude and longitude from radians to approximate meters from origin.
        # This is a simplified Cartesian conversion, more accurate methods exist for larger distances.
        self.state[0] *= RADIUS  # Latitude (becomes ~Y in a local ENU frame if origin is equator/prime meridian)
        self.state[1] *= RADIUS  # Longitude (becomes ~X)
        # self.state[2] is altitude in meters, no change needed.

        # Concatenate aircraft state and goal position.
        return np.hstack([self.state, self.goal]).astype(np.float32)

    def step(self, action: np.ndarray):
        """
        Applies an action to the environment and steps the simulation forward.

        Args:
            action (np.ndarray): The action to apply, conforming to `self.action_space`.
                                 [roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd].

        Returns:
            tuple: A tuple containing:
                - obs (np.ndarray): The stacked observation after the step.
                - reward (float): The reward received for this step.
                - terminated (bool): True if the episode ended due to a terminal state (crash/goal).
                - truncated (bool): True if the episode ended due to a time limit or other truncation.
                - info (dict): Auxiliary diagnostic information (currently empty).
        """
        self.current_step += 1
        roll_cmd, pitch_cmd, yaw_cmd, throttle = action

        # Set JSBSim flight control properties from the action.
        self.simulation.set_property_value("fcs/aileron-cmd-norm", float(roll_cmd))
        self.simulation.set_property_value("fcs/elevator-cmd-norm", float(pitch_cmd))
        self.simulation.set_property_value("fcs/rudder-cmd-norm", float(yaw_cmd))
        self.simulation.set_property_value("fcs/throttle-cmd-norm", float(throttle))

        # Run multiple JSBSim simulation steps for each environment step.
        for _ in range(self.down_sample):
            # Ensure fuel tanks are not empty (simplified fuel handling).
            self.simulation.set_property_value("propulsion/tank/contents-lbs", 1000.0)
            self.simulation.set_property_value("propulsion/tank[1]/contents-lbs", 1000.0)
            # Ensure landing gear is up.
            self.simulation.set_property_value("gear/gear-cmd-norm", 0.0)
            self.simulation.set_property_value("gear/gear-pos-norm", 0.0)
            self.simulation.run()

        current_single_obs = self._get_current_single_observation()
        self.obs_buffer.append(current_single_obs) # Add to observation stack.

        reward = 0.0
        terminated = False
        truncated = False

        # Check for termination conditions.
        latest_aircraft_state = current_single_obs[:12] # Aircraft part of the current obs
        altitude_m = latest_aircraft_state[2]
        # Crash condition: altitude below 10 meters.
        if altitude_m < 10.0:
            reward = -10.0
            terminated = True

        # Goal reached condition: within distance `dg` of the goal in 3D.
        # latest_aircraft_state[0] is scaled latitude (approx. Y_enu), [1] is scaled longitude (approx. X_enu)
        # Goal coordinates are also in this approximate Cartesian frame.
        dist_to_goal_2d_sq = (latest_aircraft_state[0] - self.goal[0])**2 + \
                             (latest_aircraft_state[1] - self.goal[1])**2
        if not terminated and np.sqrt(dist_to_goal_2d_sq) < self.dg and \
           abs(altitude_m - self.goal[2]) < self.dg:
            reward = 10.0
            terminated = True

        # Truncation condition: maximum episode steps reached.
        if not terminated and self.current_step >= self.max_episode_steps:
            truncated = True

        obs = np.array(self.obs_buffer, dtype=np.float32)
        info = {} # Auxiliary information dictionary.

        # Internal check for observation space bounds (optional, for debugging).
        # Note: `self.observation_space.contains(obs)` is the correct Gymnasium check.
        if not self.observation_space.contains(obs):
            # Detailed print for out-of-bounds values if check fails.
            # This helps identify issues with state bounds or normalization.
            low_bound_violations = obs < self.observation_space.low
            high_bound_violations = obs > self.observation_space.high
            # Consider only finite values for bound checks, as Inf is allowed by some bounds.
            finite_obs = np.isfinite(obs)
            
            bad_low = low_bound_violations & finite_obs
            bad_high = high_bound_violations & finite_obs
            
            if np.any(bad_low) or np.any(bad_high):
                print("Warning: Observation is outside defined space bounds.")
                # Further detailed logging can be added here if needed.
            # It's generally better to let the agent handle out-of-bound observations
            # if they are rare and result from extreme states, rather than crashing.
            # However, for strict compliance or debugging, an error can be raised.
            # raise ValueError("Observation outside defined space after processing.")

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: dict = None):
        """
        Resets the environment to an initial state with a new random goal.

        Args:
            seed (int, optional): Seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options for resetting (not used currently).

        Returns:
            tuple: A tuple containing:
                - obs (np.ndarray): The initial stacked observation.
                - info (dict): Auxiliary diagnostic information (currently empty).
        """
        super().reset(seed=seed) # Sets up self.np_random if Gymnasium version expects it.
        self.current_step = 0

        self.simulation.run_ic() # Reset JSBSim to initial conditions.
        self.simulation.set_property_value('propulsion/set-running', -1) # Ensure engines are on.

        # Initialize random number generator for goal placement.
        # If super().reset(seed=seed) already initializes self.np_random:
        # rng = self.np_random
        # Else, create a new one:
        rng = np.random.default_rng(seed)

        # Generate a new random goal position.
        distance_m = rng.uniform(1000.0, 10000.0) # Distance from start (1km to 10km).
        bearing_rad = rng.uniform(0, 2 * np.pi)   # Bearing from start.
        altitude_m = rng.uniform(1000.0, 4000.0)  # Goal altitude (1km to 4km).

        # Goal coordinates in the same approximate Cartesian frame as aircraft state.
        # Assuming aircraft starts near (0,0) in this local frame.
        self.goal[0] = distance_m * np.cos(bearing_rad) # Approx X_enu
        self.goal[1] = distance_m * np.sin(bearing_rad) # Approx Y_enu
        self.goal[2] = altitude_m

        initial_single_obs = self._get_current_single_observation()
        self.obs_buffer.clear()
        # Fill the observation buffer with copies of the initial observation.
        for _ in range(self.num_stacked_frames):
            self.obs_buffer.append(np.copy(initial_single_obs))

        return np.array(self.obs_buffer, dtype=np.float32), {}

    def render(self, mode: str = 'human'):
        """
        Renders the current state of the environment.

        Args:
            mode (str): The rendering mode. Supported modes:
                        'human': Renders to a display window (if Viewer supports it).
                        'rgb_array': Returns an RGB array of the current scene.

        Returns:
            np.ndarray or None: An RGB array if mode is 'rgb_array', otherwise None
                                for 'human' mode (as rendering is to a window).
                                Returns None if viewer initialization fails.
        """
        scale = 1e-3 # Scale factor for rendering positions.

        if self.viewer is None:
            try:
                # Initialize the Viewer (headless by default in provided rendering.py)
                self.viewer = Viewer(1280, 720) # Width, Height for rendering context.

                # Load mesh for F-16 aircraft.
                # Assumes 'f16.obj' is findable relative to rendering.py or JSBSim root.
                # The program (shader) is obtained from the viewer instance.
                f16_mesh_vao = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
                self.f16_render_obj = RenderObject(f16_mesh_vao)
                self.f16_render_obj.transform.scale = 1.0 / 30.0
                self.f16_render_obj.color = (0.0, 0.0, 0.4) # Blueish
                self.viewer.objects.append(self.f16_render_obj)

                # Load mesh for the goal cylinder.
                goal_mesh_vao = load_mesh(self.viewer.ctx, self.viewer.prog, "cylinder.obj")
                self.goal_render_obj = RenderObject(goal_mesh_vao)
                self.goal_render_obj.transform.scale = scale * 100.0 # Scale for visibility
                self.goal_render_obj.color = (0.0, 0.4, 0.0) # Green
                self.viewer.objects.append(self.goal_render_obj)

                # Create and add a grid to the scene.
                # Grid uses the 'unlit' shader program from the viewer.
                self.viewer.objects.append(Grid(self.viewer.ctx, self.viewer.unlit_prog, 21, 1.0))
            except Exception as e:
                print(f"Error initializing renderer or loading assets: {e}")
                self.viewer = None # Prevent further rendering attempts if setup fails.
                return None # Indicate failure to render.

        if self.viewer is None: # Check again if initialization failed.
            return None

        # Get current aircraft state for rendering (already normalized and scaled).
        # self.state[:3] contains [scaled_lat_meters, scaled_lon_meters, alt_meters].
        # self.state[9:12] contains [phi_norm, theta_norm, psi_norm].
        aircraft_pos_sim = self.state[:3]
        aircraft_euler_sim = self.state[9:12]

        # Update F-16 visual object's transform.
        # JSBSim X (fwd) to Viewer Z, JSBSim Y (right) to Viewer -X, JSBSim Z_down (alt) to Viewer Y_up.
        self.f16_render_obj.transform.position = np.array([
            -aircraft_pos_sim[1] * scale,  # Viewer X from -SimY
            aircraft_pos_sim[2] * scale,   # Viewer Y from SimZ (altitude)
            aircraft_pos_sim[0] * scale    # Viewer Z from SimX
        ], dtype=np.float32)

        # Convert aircraft's Euler angles to Quaternion for rendering.
        q_aircraft_world = Quaternion.from_euler(*aircraft_euler_sim) # phi, theta, psi
        # Apply coordinate system transformation for visualization if necessary.
        # The original transformation was: q_display = Quaternion(q_jsb.w, -q_jsb.y, -q_jsb.z, q_jsb.x)
        # This depends on the .obj model's native orientation and viewer's camera setup.
        # For now, using the provided transformation.
        self.f16_render_obj.transform.rotation = Quaternion(
            q_aircraft_world.w, -q_aircraft_world.y, -q_aircraft_world.z, q_aircraft_world.x
        )

        # Update goal visual object's transform.
        goal_pos_sim = self.goal
        self.goal_render_obj.transform.position = np.array([
            -goal_pos_sim[1] * scale, # Viewer X
            goal_pos_sim[2] * scale,  # Viewer Y
            goal_pos_sim[0] * scale   # Viewer Z
        ], dtype=np.float32)
        # Goal typically has a fixed orientation (e.g., identity quaternion).
        self.goal_render_obj.transform.rotation = Quaternion() # Identity

        # Set camera view based on aircraft and goal positions.
        # This logic aims to position the camera behind the aircraft, looking towards the goal vicinity.
        f16_viewer_pos = self.f16_render_obj.transform.position
        goal_viewer_pos = self.goal_render_obj.transform.position
        
        vec_goal_to_f16 = f16_viewer_pos - goal_viewer_pos
        dist_goal_to_f16 = np.linalg.norm(vec_goal_to_f16)
        if dist_goal_to_f16 > 1e-6:
            dir_goal_to_f16 = vec_goal_to_f16 / dist_goal_to_f16
        else:
            dir_goal_to_f16 = np.array([0, 0, -1], dtype=np.float32) # Default if coincident

        # Camera positioned behind F16, slightly elevated, looking in general direction of F16.
        # The original set_view took: *(r + cylinder_pos + rhat + offset), Quaternion.from_euler(...)
        # where r = f16_pos - cylinder_pos.
        # This means cam_eye = f16_pos + rhat + offset_y.
        # And orientation was derived from r.
        cam_eye_position = f16_viewer_pos + dir_goal_to_f16 * 1.5 + np.array([0, 0.5, 0], dtype=np.float32) # Example offset

        # Calculate yaw and pitch for camera to look from cam_eye_position towards f16_viewer_pos
        look_direction = f16_viewer_pos - cam_eye_position
        dist_look = np.linalg.norm(look_direction)
        if dist_look > 1e-6:
            look_direction /= dist_look
        
        # Yaw around global Y, Pitch around camera's local X.
        # Assuming viewer +Y is up, +X is right, +Z is "out of screen" (left-handed view) or "into screen" (right-handed)
        # The original used: yaw = np.arctan2(-x,-z); pitch = np.arctan2(-y, sqrt(x^2+z^2)) from r.
        # Let r_cam = -look_direction (vector from eye to target)
        r_cam_x, r_cam_y, r_cam_z = -look_direction[0], -look_direction[1], -look_direction[2]
        cam_yaw = np.arctan2(r_cam_x, r_cam_z) # Yaw from X and Z components (if Z is fwd/back)
        cam_pitch = np.arctan2(r_cam_y, np.sqrt(r_cam_x**2 + r_cam_z**2)) # Pitch from Y and XZ plane distance
        
        # The original call used -pitch for from_euler. mode=1 for Quaternion.from_euler might be 'ZYX' or specific.
        self.viewer.set_view(
            x=cam_eye_position[0], y=cam_eye_position[1], z=cam_eye_position[2],
            rotation_quaternion=Quaternion.from_euler(-cam_pitch, cam_yaw, 0, mode=1) # Original angles, mode=1 for quat
        )

        # Call the viewer's main render method (which draws to its internal FBO if headless)
        self.viewer.render() # This is Viewer.render(), not this JSBSimEnv.render()

        if mode == 'rgb_array':
            return self.viewer.get_frame()
        # For 'human' mode, if Viewer is windowed, render() would have flipped display.
        # If Viewer is headless, human mode might not show anything or could also return get_frame().
        # Current Viewer is headless by default, so 'human' mode won't open a window from here.
        return None # Or self.viewer.get_frame() if 'human' should also yield array for headless

    def close(self):
        """Cleans up resources, including closing the viewer if it was initialized."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class PositionReward(gym.Wrapper):
    """
    A Gymnasium wrapper that adds potential-based reward shaping to the JSBSimEnv.

    The agent receives an additional reward proportional to the change in distance
    to the goal: positive for moving closer, negative for moving farther.
    This helps in alleviating sparse reward issues for navigation tasks.

    Args:
        env: The JSBSimEnv instance to wrap.
        gain (float): The scaling factor for the potential-based reward.
    """
    def __init__(self, env: JSBSimEnv, gain: float):
        super().__init__(env)
        self.gain = gain
        self.last_distance = 0.0 # Initialized in reset

    def step(self, action: np.ndarray):
        """
        Steps the wrapped environment and modifies the reward based on distance to goal.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_single_obs = obs[-1] # Latest frame from the stacked observation
        # Aircraft position (X, Y, Alt in meters) is in the first 3 components.
        # Goal position (X, Y, Alt in meters) is in the last 3 components.
        aircraft_pos = current_single_obs[:3]
        goal_pos = current_single_obs[-3:] # These are identical to self.env.goal (or self.goal if accessed directly)

        displacement_vec = goal_pos - aircraft_pos
        current_distance = np.linalg.norm(displacement_vec)

        # Potential-based reward: gain * (previous_potential - current_potential)
        # Here, potential is -distance (we want to minimize distance).
        # So, reward is gain * (-last_distance - (-current_distance)) = gain * (current_distance - last_distance)
        # Or, if potential is distance, reward is gain * (last_distance - current_distance) for moving closer.
        reward += self.gain * (self.last_distance - current_distance)
        self.last_distance = current_distance

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs): # Pass through kwargs like seed, options
        """Resets the wrapped environment and initializes the last_distance."""
        obs, info = self.env.reset(**kwargs)
        current_single_obs = obs[-1]
        aircraft_pos = current_single_obs[:3]
        goal_pos = current_single_obs[-3:]
        displacement_vec = goal_pos - aircraft_pos
        self.last_distance = np.linalg.norm(displacement_vec)
        return obs, info

def wrap_jsbsim(**kwargs) -> PositionReward:
    """
    Factory function to create an instance of JSBSimEnv wrapped with PositionReward.

    Args:
        **kwargs: Arguments to pass to the JSBSimEnv constructor (e.g., `root`).

    Returns:
        PositionReward: The wrapped JSBSim environment.
    """
    env = JSBSimEnv(**kwargs)
    wrapped_env = PositionReward(env, gain=1e-2) # Default gain for reward shaping
    return wrapped_env

# Register the environment with Gymnasium.
# This allows gym.make("JSBSim-v0") to create the wrapped environment.
gym.register(
    id="JSBSim-v0",
    entry_point="jsbsim_gym.nav_env:wrap_jsbsim", # Assumes this file is jsbsim_gym.py within jsbsim_gym package
    # If this file is top-level, e.g. 'my_env_file.py', entry_point would be 'my_env_file:wrap_jsbsim'
    max_episode_steps=1200, # Used by TimeLimit wrapper if not applied manually
    # Additional arguments for wrap_jsbsim can be passed via kwargs in gym.make
    # or set as defaults here:
    # kwargs={'root': '.'} # Example
)

# Example usage block for testing the environment directly.
if __name__ == "__main__":
    from time import sleep
    print("Creating and testing JSBSim-v0 environment...")
    # The entry_point in gym.register should be correct for this import path.
    # If this file is 'jsbsim_gym/jsbsim_gym.py', then
    # 'jsbsim_gym.jsbsim_gym:wrap_jsbsim' is correct.
    env = gym.make("JSBSim-v0") # `root` will default to '.' in JSBSimEnv

    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    obs, info = env.reset(seed=42)
    print("Initial stacked observation shape:", obs.shape)
    # print("Initial single obs (last frame):", obs[-1]) # For debugging
    
    # Initial render call to set up viewer
    try:
        frame_data = env.render(mode='rgb_array') # Request rgb_array for testing
        if frame_data is not None:
            print(f"Rendered initial frame, shape: {frame_data.shape}")
        else:
            print("Initial render call returned None.")
    except Exception as e:
        print(f"Error during initial render call: {e}")


    for i in range(env.spec.max_episode_steps + 50 if env.spec else 1250):
        action = env.action_space.sample() # Sample a random action.

        try:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        except ValueError as e_step: # Catch potential "Observation outside space"
            print(f"ValueError during step {i}: {e_step}")
            break # Stop if environment behaves unexpectedly

        if i % 10 == 0:
            print(f"Step {i}, Reward: {reward:.3f}, Term: {terminated}, Trunc: {truncated}, Done: {done}")
        
        try:
            frame_data = env.render(mode='rgb_array') # Request rgb_array for testing
            # if frame_data is None and i % 10 == 0 :
            #     print(f"Warning: render() returned None at step {i}")
        except Exception as e_render:
            print(f"Error during render call at step {i}: {e_render}")
            # Decide if to break or continue if rendering fails
            # break

        if done:
            print(f"Episode finished after {i+1} timesteps. Terminated: {terminated}, Truncated: {truncated}")
            obs, info = env.reset(seed=i + 100) # Use a different seed for subsequent resets
            print("Environment reset. New obs shape:", obs.shape)
        
        sleep(1/60) # Slow down for observation if rendering a window, or for controlled headless runs.
                    # For pure headless data generation, this sleep can be removed.
    
    print("Closing environment.")
    env.close()
    print("Test finished.")