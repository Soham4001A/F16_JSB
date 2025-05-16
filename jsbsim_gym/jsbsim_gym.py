import jsbsim
import gym
import numpy as np
from os import path # For loading meshes in render

# ---------------------------------------------------------------------------
# Temporary compatibility patch for Gym / Numpy 2.0:
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
# ---------------------------------------------------------------------------

from .visualization.rendering import Viewer, load_mesh, load_shader, RenderObject, Grid
from .visualization.quaternion import Quaternion

# Initialize format for the environment state vector (AGENT STATE)
AGENT_STATE_FORMAT = [
    "position/lat-gc-rad",
    "position/long-gc-rad",
    "position/h-sl-meters",
    "velocities/mach",
    "aero/alpha-rad",
    "aero/beta-rad",
    "velocities/p-rad_sec",
    "velocities/q-rad_sec",
    "velocities/r-rad_sec",
    "attitude/phi-rad",
    "attitude/theta-rad",
    "attitude/psi-rad",
]

# Observation space bounds: [agent_state (12), goal_position (3)]
OBSERVATION_LOW = np.array([
    -np.inf, -np.inf, 0,      # Lat, Lon, Alt
    0, -np.pi, -np.pi,        # Mach, Alpha, Beta
    -np.inf, -np.inf, -np.inf,# P, Q, R
    -np.pi, -np.pi, -np.pi,   # Phi, Theta, Psi
    -np.inf, -np.inf, 0,      # Goal X, Y, Z (or Boom X, Y, Z)
], dtype=np.float32)

OBSERVATION_HIGH = np.array([
    np.inf, np.inf, np.inf,
    np.inf, np.pi, np.pi,
    np.inf, np.inf, np.inf,
    np.pi, np.pi, np.pi,
    np.inf, np.inf, np.inf,
], dtype=np.float32)

# Action space: Roll, Pitch, Yaw commands (normalized), Throttle command
ACTION_LOW = np.array([-1, -1, -1, 0], dtype=np.float32)
ACTION_HIGH = np.array([1, 1, 1, 1], dtype=np.float32) # Max throttle is 1.0

# Radius of the earth
RADIUS = 6.3781e6

class JSBSimEnvBase(gym.Env):
    """
    Base class for JSBSim environments. Handles common JSBSim setup,
    agent state updates, and basic rendering infrastructure.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, root='.', jsbsim_model='f16', down_sample_rate=4):
        super().__init__()

        self.observation_space = gym.spaces.Box(OBSERVATION_LOW, OBSERVATION_HIGH, (15,), dtype=np.float32)
        self.action_space = gym.spaces.Box(ACTION_LOW, ACTION_HIGH, (4,), dtype=np.float32)

        self.simulation = jsbsim.FGFDMExec(root, None)
        self.simulation.set_debug_level(0)
        self.jsbsim_model_name = jsbsim_model
        self.simulation.load_model(self.jsbsim_model_name)
        
        self._set_initial_conditions()
        self.simulation.run_ic() # Apply initial conditions

        self.down_sample = down_sample_rate # JSBSim steps per env step
        self.agent_state = np.zeros(len(AGENT_STATE_FORMAT)) # Stores agent's state

        self.viewer = None
        self.render_scale = 1e-3 # For visualization scaling

        # Render objects common to all or most envs
        self.agent_render_obj = None
        self.grid_render_obj = None
        
        # Path to object files (assuming they are in the same dir as this script or accessible)
        # Adjust this path if your .obj files are elsewhere
        self.obj_file_path = path.join(path.abspath(path.dirname(__file__)), 'visualization', 'meshes')


    def _set_initial_conditions(self):
        # Set engines running, forward velocity, and altitude for the agent
        self.simulation.set_property_value('propulsion/set-running', -1)
        self.simulation.set_property_value('ic/u-fps', 900.) # ~274 m/s or Mach 0.8 at 5000ft
        self.simulation.set_property_value('ic/h-sl-ft', 5000)
        self.simulation.set_property_value('ic/phi-deg', 0)
        self.simulation.set_property_value('ic/theta-deg', 0)
        self.simulation.set_property_value('ic/psi-deg', 0)


    def _get_agent_state(self):
        # Gather all agent state properties from JSBSim
        for i, prop_name in enumerate(AGENT_STATE_FORMAT):
            self.agent_state[i] = self.simulation.get_property_value(prop_name)
        
        # Rough conversion of lat/lon to meters (relative to sim start, assumed near 0,0)
        self.agent_state[0] *= RADIUS  # Lat to Y-ish (careful with interpretation)
        self.agent_state[1] *= RADIUS  # Lon to X-ish
        # Note: JSBSim's lat/lon are geocentric. For local flat-earth approx:
        # x_local = R * (lon - lon_ref) * cos(lat_ref)
        # y_local = R * (lat - lat_ref)
        # For simplicity here, we assume lat_ref=0 and treat converted values as local Cartesian.
        # Position is [lat, lon, alt] -> effectively [y_sim, x_sim, z_sim] after scaling by RADIUS

    def _apply_action_to_jsbsim(self, action):
        roll_cmd, pitch_cmd, yaw_cmd, throttle = action
        self.simulation.set_property_value("fcs/aileron-cmd-norm", roll_cmd)
        self.simulation.set_property_value("fcs/elevator-cmd-norm", pitch_cmd)
        self.simulation.set_property_value("fcs/rudder-cmd-norm", yaw_cmd)
        self.simulation.set_property_value("fcs/throttle-cmd-norm", throttle)

    def _run_jsbsim_steps(self):
        for _ in range(self.down_sample):
            # Freeze fuel consumption
            self.simulation.set_property_value("propulsion/tank/contents-lbs", 1000)
            self.simulation.set_property_value("propulsion/tank[1]/contents-lbs", 1000)
            # Set gear up
            self.simulation.set_property_value("gear/gear-cmd-norm", 0.0)
            self.simulation.set_property_value("gear/gear-pos-norm", 0.0)
            if not self.simulation.run():
                # print("JSBSim FAILED to run a step!") # Should handle this failure
                return False # Indicate failure
        return True # Indicate success

    def _setup_viewer(self):
        if self.viewer is None:
            self.viewer = Viewer(1280, 720, "JSBSim Environment")
            agent_mesh_path = path.join(self.obj_file_path, "f16.obj")
            agent_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, agent_mesh_path)
            if agent_mesh:
                self.agent_render_obj = RenderObject(agent_mesh)
                self.agent_render_obj.transform.scale = 1/30 
                self.agent_render_obj.color = 0.1, 0.1, 0.8 # Blueish agent
                self.viewer.objects.append(self.agent_render_obj)
            
            self.grid_render_obj = Grid(self.viewer.ctx, self.viewer.unlit, 21, 1000. * self.render_scale) # Grid lines every km
            self.viewer.objects.append(self.grid_render_obj)

    def _render_agent(self):
        if self.agent_render_obj:
            # JSBSim state: lat(rad), lon(rad), alt(m), ..., phi, theta, psi (rad)
            # Agent state: [y_sim, x_sim, z_sim] (approx after RADIUS scaling)
            # Render mapping: z_render = x_sim, x_render = -y_sim, y_render = z_sim
            
            x_sim, y_sim, z_sim = self.agent_state[1], self.agent_state[0], self.agent_state[2] # lon, lat, alt
            
            self.agent_render_obj.transform.x = -y_sim * self.render_scale
            self.agent_render_obj.transform.y = z_sim * self.render_scale
            self.agent_render_obj.transform.z = x_sim * self.render_scale


            # JSBSim attitude: phi (roll, around X_body), theta (pitch, around Y_body), psi (yaw, around Z_body)
            # Our Quaternion.from_euler expects (yaw, pitch, roll) by default if mode='zyx'
            # Or (roll, pitch, yaw) if mode not specified (defaulting to intrinsic Tait-Bryan ZYX: psi, theta, phi)
            # Let's use the JSBSim order: phi, theta, psi directly
            phi, theta, psi = self.agent_state[9], self.agent_state[10], self.agent_state[11]
            
            # The original visualization code used:
            # rot = Quaternion.from_euler(*self.state[9:]) -> phi, theta, psi
            # rot = Quaternion(rot.w, -rot.y, -rot.z, rot.x)
            # This implies a coordinate system transformation:
            # JSBSim body: X_fwd, Y_right, Z_down
            # Visualization seems to map to: Z_fwd_render, X_right_render, Y_up_render
            # Original Euler angles (phi, theta, psi) are rotations about X_body, Y_body, Z_body
            q_jsbsim_att = Quaternion.from_euler(phi, theta, psi) # Default is intrinsic ZYX (psi,theta,phi)
                                                               # If input is (phi,theta,psi) direct, it should be fine for from_euler.
                                                               # Need to be careful with Euler sequence. JSBSim is typically phi,theta,psi.
            
            # Let's assume from_euler takes (roll, pitch, yaw) in BODY frame
            # JSBSim outputs phi (roll), theta (pitch), psi (yaw)
            q_body_to_inertial = Quaternion.from_euler(phi, theta, psi) # Order: roll, pitch, yaw

            # Transform from JSBSim's NED body frame (X fwd, Y right, Z down)
            # to visualization's frame (e.g., X right, Y up, Z backward/forward)
            # If render is X_right_viz, Y_up_viz, Z_fwd_viz (OpenGL like, but Z positive forward)
            # and JSBSim is X_fwd_JSB, Y_right_JSB, Z_down_JSB
            # A rotation from JSB_body to VIZ_body:
            # X_viz = Y_JSB
            # Y_viz = -Z_JSB
            # Z_viz = X_JSB
            # This is q_viz_from_jsb = Quaternion.from_euler(pi/2, 0, pi/2)
            # Then total_rotation = q_body_to_inertial * q_viz_from_jsb
            # The old conversion: Quaternion(rot.w, -rot.y, -rot.z, rot.x)
            # suggests: w'=w, x'=-y, y'=-z, z'=x. This is a specific axis swap.
            self.agent_render_obj.transform.rotation = Quaternion(q_body_to_inertial.w, 
                                                                  -q_body_to_inertial.y, # JSB Y (pitch) maps to -Viz X
                                                                  -q_body_to_inertial.z, # JSB Z (yaw) maps to -Viz Y
                                                                   q_body_to_inertial.x)  # JSB X (roll) maps to Viz Z


    def render(self, mode='human'):
        raise NotImplementedError("Subclasses must implement render specific to their scenario.")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for seeding the random number generator
        
        # Rerun initial conditions in JSBSim for the agent
        # self.simulation.load_model(self.jsbsim_model_name) # Reload to ensure clean state
        self._set_initial_conditions()
        self.simulation.run_ic()
        self.simulation.set_property_value('propulsion/set-running', -1) # Ensure engines are on
        
        self._get_agent_state() # Get initial agent state
        
        # Derived classes will handle goal/tanker setup and return obs, info
        return {}, {} # Placeholder, must be overridden


class JSBSimEnv(JSBSimEnvBase):
    """
    Original point-to-point navigation task.
    The agent must navigate to a static cylindrical goal region.
    """
    def __init__(self, root='.', jsbsim_model='f16', down_sample_rate=4):
        super().__init__(root, jsbsim_model, down_sample_rate)
        self.goal_world_pos = np.zeros(3) # X, Y, Z in world frame
        self.dg = 100  # Success radius in meters for reaching goal

        # Render specific objects
        self.goal_render_obj = None

    def _setup_viewer_scene_specific(self):
        if self.viewer and self.goal_render_obj is None:
            goal_mesh_path = path.join(self.obj_file_path, "cylinder.obj")
            goal_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, goal_mesh_path)
            if goal_mesh:
                self.goal_render_obj = RenderObject(goal_mesh)
                self.goal_render_obj.transform.scale = self.dg * self.render_scale # Scale to dg
                self.goal_render_obj.color = 0, .4, 0 # Green goal
                self.viewer.objects.append(self.goal_render_obj)

    def _update_render_scene_specific(self):
        if self.goal_render_obj:
            # Goal position is [X_world, Y_world, Z_world]
            # Mapping to render: z_render = X_world, x_render = -Y_world, y_render = Z_world
            self.goal_render_obj.transform.x = -self.goal_world_pos[1] * self.render_scale
            self.goal_render_obj.transform.y =  self.goal_world_pos[2] * self.render_scale
            self.goal_render_obj.transform.z =  self.goal_world_pos[0] * self.render_scale

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handles agent ICs and gets agent_state
        rng = np.random.default_rng(self.np_random.bit_generator) # Use Gym's seeded RNG

        # Generate a new static goal
        # Goal relative to agent's initial world position (which is effectively origin due to conversion)
        distance = rng.uniform(1000, 10000) # 1km to 10km
        bearing_rad = rng.uniform(0, 2 * np.pi)
        # Target altitude relative to ground, or absolute. Let's make it absolute for simplicity.
        altitude_m = rng.uniform(self.agent_state[2] - 1000, self.agent_state[2] + 1000)
        altitude_m = np.clip(altitude_m, 500, 6000) # Keep goal within reasonable flight envelope

        # Convert bearing/distance to X, Y (assuming agent starts at ~origin facing ~positive X due to psi=0)
        # Goal X (forward/backward), Y (left/right), Z (altitude)
        self.goal_world_pos[0] = self.agent_state[1] + distance * np.cos(bearing_rad) # Goal X (matches lon-derived x_sim)
        self.goal_world_pos[1] = self.agent_state[0] + distance * np.sin(bearing_rad) # Goal Y (matches lat-derived y_sim)
        self.goal_world_pos[2] = altitude_m
        
        obs = np.hstack([self.agent_state, self.goal_world_pos]).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._apply_action_to_jsbsim(action)
        if not self._run_jsbsim_steps():
            # JSBSim failed, treat as a catastrophic failure
            obs = np.hstack([self.agent_state, self.goal_world_pos]).astype(np.float32)
            return obs, -500, True, False, {"jsbsim_error": True} 

        self._get_agent_state()

        reward = 0
        terminated = False
        truncated = False

        # Check for collision with ground
        if self.agent_state[2] < 10: # Agent altitude
            reward = -100 # Increased penalty
            terminated = True

        # Check if reached goal
        current_displacement_vec = self.goal_world_pos - self.agent_state[[1,0,2]] # Goal(X,Y,Z) - Agent(X_sim, Y_sim, Z_sim)
        distance_to_goal_xy = np.linalg.norm(current_displacement_vec[:2])
        distance_to_goal_z = abs(current_displacement_vec[2])

        if distance_to_goal_xy < self.dg and distance_to_goal_z < self.dg :
            reward = 100 # Increased reward
            terminated = True
        
        obs = np.hstack([self.agent_state, self.goal_world_pos]).astype(np.float32)
        return obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        self._setup_viewer() # From base
        self._setup_viewer_scene_specific() # For goal cylinder

        self._render_agent() # From base
        self._update_render_scene_specific() # For goal cylinder

        # Simplified chase camera on agent for point-to-point
        if self.viewer and self.agent_render_obj:
            agent_pos_viz = self.agent_render_obj.transform.position
            agent_rot_viz = self.agent_render_obj.transform.rotation
            
            # Camera behind and slightly above the agent
            offset_cam_body = np.array([0, 1.0, -5.0]) # X_body_viz (right), Y_body_viz (up), Z_body_viz (fwd)
                                                       # So this is 1m up, 5m behind in agent's viz frame
            
            cam_pos_offset_world = agent_rot_viz.rotate_vector(offset_cam_body)
            eye = agent_pos_viz + cam_pos_offset_world
            
            # Look slightly ahead of the agent
            look_at_offset_body = np.array([0, 0.5, 10.0]) 
            center = agent_pos_viz + agent_rot_viz.rotate_vector(look_at_offset_body)
            
            up_body = np.array([0, 1, 0]) # Y-up in body frame
            up_world = agent_rot_viz.rotate_vector(up_body)
            
            self.viewer.set_view_matrix(eye, center, up_world)

        if self.viewer:
            self.viewer.render()
            if mode == 'rgb_array':
                return self.viewer.get_frame()
        return None


class JSBSimTankerEnv(JSBSimEnvBase):
    """
    Air-to-air refueling task. The agent must navigate to and maintain
    position relative to a moving boom on a tanker aircraft.
    """
    def __init__(self, root='.', jsbsim_model='f16', down_sample_rate=4):
        super().__init__(root, jsbsim_model, down_sample_rate)
        
        self.tanker_initial_speed_mps = self.simulation.get_property_value('ic/u-fps') * 0.3048 # Match agent's initial speed in m/s
        self.tanker_state = np.zeros(7)  # [x_world, y_world, z_world, psi_world, theta_world, phi_world, speed_mps]
        
        # Boom offset from tanker CG in tanker's BODY frame (X_fwd, Y_right, Z_down)
        self.boom_offset_body = np.array([-17.0, -3.0, -1.5]) # Meters: 17m aft, 3m left (port), 1.5m down
        self.boom_wobble_std_dev = 0.05 # Meters, for random boom movement
        self.boom_world_pos = np.zeros(3) # Current world position of the boom tip

        self.dg = 2.0  # Success radius for boom "contact" in meters

        # Render specific objects
        self.tanker_render_obj = None
        self.boom_render_obj = None


    def _setup_viewer_scene_specific(self):
        if self.viewer:
            if self.tanker_render_obj is None:
                tanker_mesh_path = path.join(self.obj_file_path, "f16.obj") # Use F16 for tanker for now
                tanker_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, tanker_mesh_path)
                if tanker_mesh:
                    self.tanker_render_obj = RenderObject(tanker_mesh)
                    self.tanker_render_obj.transform.scale = 1/30 
                    self.tanker_render_obj.color = 0.8, 0.1, 0.1 # Reddish tanker
                    self.viewer.objects.append(self.tanker_render_obj)

            if self.boom_render_obj is None:
                boom_mesh_path = path.join(self.obj_file_path, "sphere.obj") # Simple sphere for boom tip
                if not path.exists(boom_mesh_path): # Fallback
                    boom_mesh_path = path.join(self.obj_file_path, "cylinder.obj")
                boom_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, boom_mesh_path)
                if boom_mesh:
                    self.boom_render_obj = RenderObject(boom_mesh)
                    # Scale based on dg for visibility, or fixed small size
                    self.boom_render_obj.transform.scale = self.dg * self.render_scale * 0.5 
                    self.boom_render_obj.color = 0.1, 0.8, 0.1 # Green boom
                    self.viewer.objects.append(self.boom_render_obj)

    def _update_render_scene_specific(self):
        # Tanker rendering
        if self.tanker_render_obj:
            # Tanker state: [x_w, y_w, z_w, psi_w, theta_w, phi_w, speed]
            # Mapping to render: z_render = x_w, x_render = -y_w, y_render = z_w
            self.tanker_render_obj.transform.x = -self.tanker_state[1] * self.render_scale
            self.tanker_render_obj.transform.y =  self.tanker_state[2] * self.render_scale
            self.tanker_render_obj.transform.z =  self.tanker_state[0] * self.render_scale
            
            phi_t, theta_t, psi_t = self.tanker_state[5], self.tanker_state[4], self.tanker_state[3]
            q_tanker_att = Quaternion.from_euler(phi_t, theta_t, psi_t)
            self.tanker_render_obj.transform.rotation = Quaternion(q_tanker_att.w,
                                                                   -q_tanker_att.y,
                                                                   -q_tanker_att.z,
                                                                    q_tanker_att.x)
        # Boom rendering
        if self.boom_render_obj:
            self.boom_render_obj.transform.x = -self.boom_world_pos[1] * self.render_scale
            self.boom_render_obj.transform.y =  self.boom_world_pos[2] * self.render_scale
            self.boom_render_obj.transform.z =  self.boom_world_pos[0] * self.render_scale


    def _body_to_world_rotation_matrix(self, phi, theta, psi):
        # Standard ZYX Euler sequence: R = Rz(psi)Ry(theta)Rx(phi)
        cph, sph = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cps, sps = np.cos(psi), np.sin(psi)

        Rx = np.array([[1, 0, 0], [0, cph, -sph], [0, sph, cph]])
        Ry = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
        Rz = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])
        
        # For body-to-world, R_bw = (R_wb)^T. R_wb is RzRyRx for ZYX convention.
        # So, R_body_to_world = (Rz @ Ry @ Rx)
        # This matrix transforms a vector from body coordinates to world coordinates.
        return Rz @ Ry @ Rx


    def _update_tanker_kinematics(self, dt):
        # Tanker state: [x_w, y_w, z_w, psi_w, theta_w, phi_w, speed_mps]
        # For simplicity, tanker flies straight and level. psi, theta, phi are constant.
        # Velocity in world frame from body frame speed (assuming speed is along X_body)
        speed = self.tanker_state[6]
        phi, theta, psi = self.tanker_state[5], self.tanker_state[4], self.tanker_state[3]

        # Velocity components in body frame (X_fwd, Y_right, Z_down)
        v_body = np.array([speed, 0, 0]) 
        
        R_bw = self._body_to_world_rotation_matrix(phi, theta, psi)
        v_world = R_bw @ v_body
        
        self.tanker_state[0] += v_world[0] * dt # Update X_world
        self.tanker_state[1] += v_world[1] * dt # Update Y_world
        self.tanker_state[2] += v_world[2] * dt # Update Z_world

    def _update_boom_world_position(self, rng_for_wobble):
        tanker_pos_world = self.tanker_state[:3]
        phi_t, theta_t, psi_t = self.tanker_state[5], self.tanker_state[4], self.tanker_state[3]

        R_bw = self._body_to_world_rotation_matrix(phi_t, theta_t, psi_t)
        boom_offset_world = R_bw @ self.boom_offset_body
        
        base_boom_pos_world = tanker_pos_world + boom_offset_world
        
        # Add wobble (isotropic Gaussian noise in world frame)
        wobble = rng_for_wobble.normal(0, self.boom_wobble_std_dev, 3)
        self.boom_world_pos = base_boom_pos_world + wobble

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handles agent ICs and gets agent_state
        rng = np.random.default_rng(self.np_random.bit_generator)

        # Tanker Initialization (relative to agent's start)
        # Agent starts at self.agent_state[1] (x_sim_lon), self.agent_state[0] (y_sim_lat), self.agent_state[2] (z_alt)
        # with psi=0 (heading along positive x_sim_lon axis)
        
        # Tanker starts ahead of the agent
        self.tanker_state[0] = self.agent_state[1] + rng.uniform(1000, 1500)  # X_world (forward)
        self.tanker_state[1] = self.agent_state[0] + rng.uniform(-200, 200)   # Y_world (sideways)
        self.tanker_state[2] = self.agent_state[2] + rng.uniform(-50, 50)     # Z_world (altitude)
        
        self.tanker_state[3] = self.agent_state[11] # psi_world (initial heading same as agent for simplicity)
        self.tanker_state[4] = 0.0  # theta_world (pitch) - level flight
        self.tanker_state[5] = 0.0  # phi_world (roll) - level flight
        self.tanker_state[6] = self.tanker_initial_speed_mps # speed_mps

        self._update_boom_world_position(rng) # Calculate initial boom_world_pos
        
        obs = np.hstack([self.agent_state, self.boom_world_pos]).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._apply_action_to_jsbsim(action)
        
        # Calculate dt for kinematic updates
        # JSBSim's dt is fixed per internal step. Total dt for env step:
        dt_env = self.down_sample * self.simulation.get_delta_t() 

        if not self._run_jsbsim_steps():
            obs = np.hstack([self.agent_state, self.boom_world_pos]).astype(np.float32)
            return obs, -500, True, False, {"jsbsim_error": True}

        self._get_agent_state() # Update agent's state

        # Update Tanker and Boom
        self._update_tanker_kinematics(dt_env)
        # Use a non-seeded (or differently seeded) RNG for wobble to ensure it's dynamic
        self._update_boom_world_position(np.random.default_rng()) 

        reward = 0
        terminated = False
        truncated = False

        # Check for agent collision with ground
        if self.agent_state[2] < 10: # Agent altitude
            reward = -200 # Higher penalty for crashing in this critical task
            terminated = True

        # Check if agent reached/is maintaining boom position
        # Agent position in world-like frame: [x_sim_lon, y_sim_lat, z_alt]
        # self.agent_state[[1,0,2]]
        current_displacement_vec = self.boom_world_pos - self.agent_state[[1,0,2]]
        distance_to_boom = np.linalg.norm(current_displacement_vec)

        if distance_to_boom < self.dg:
            reward += 50 # Substantial reward for being in contact envelope
            # For station keeping, terminated is FALSE here.
        
        # Optional: Penalty for being too far from the tanker to encourage staying in formation area
        # distance_to_tanker = np.linalg.norm(self.tanker_state[:3] - self.agent_state[[1,0,2]])
        # max_allowable_distance_from_tanker = 3000 # meters
        # if distance_to_tanker > max_allowable_distance_from_tanker:
        #     reward -= 20 
        #     truncated = True # End episode if too far, not a crash but out of scenario bounds

        obs = np.hstack([self.agent_state, self.boom_world_pos]).astype(np.float32)
        return obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        self._setup_viewer() # From base
        self._setup_viewer_scene_specific() # For tanker and boom

        self._render_agent() # From base
        self._update_render_scene_specific() # For tanker and boom

        # Camera: Focus on the agent, try to keep boom/tanker in view
        if self.viewer and self.agent_render_obj:
            agent_pos_viz = self.agent_render_obj.transform.position
            agent_rot_viz = self.agent_render_obj.transform.rotation
            
            # Offset behind and above the agent in its own visual frame
            offset_cam_body = np.array([0, 1.5, -7.0]) # x_right, y_up, z_fwd relative to agent's viz orientation
            cam_pos_offset_world = agent_rot_viz.rotate_vector(offset_cam_body)
            eye = agent_pos_viz + cam_pos_offset_world
            
            # Look towards a point slightly ahead of the agent, or towards the boom
            # For refueling, looking towards the boom might be better if it's close
            if self.boom_render_obj:
                boom_pos_viz = self.boom_render_obj.transform.position
                # Blend look_at point between agent fwd and boom
                dir_to_boom = boom_pos_viz - agent_pos_viz
                dist_to_boom_viz = np.linalg.norm(dir_to_boom)
                
                look_at_fwd_agent_body = np.array([0,0.5,10]) # Point 10m "visually" forward of agent
                look_at_fwd_agent_world = agent_pos_viz + agent_rot_viz.rotate_vector(look_at_fwd_agent_body)

                if dist_to_boom_viz < 50 * self.render_scale : # If boom is within 50m (scaled)
                    # Prioritize looking at boom when close
                    center = boom_pos_viz 
                else:
                    center = look_at_fwd_agent_world
            else: # Fallback if boom not rendered
                look_at_fwd_agent_body = np.array([0,0.5,10])
                center = agent_pos_viz + agent_rot_viz.rotate_vector(look_at_fwd_agent_body)

            up_body = np.array([0, 1, 0]) # Y-up in agent's visual body frame
            up_world = agent_rot_viz.rotate_vector(up_body)
            
            self.viewer.set_view_matrix(eye, center, up_world)

        if self.viewer:
            self.viewer.render()
            if mode == 'rgb_array':
                return self.viewer.get_frame()
        return None


class PositionReward(gym.Wrapper):
    """
    Adds a shaping reward based on distance to the goal/boom.
    The agent is rewarded for moving closer and penalized for moving away.
    """
    def __init__(self, env, gain=1e-3, contact_bonus_on_wrapper=0.0):
        super().__init__(env)
        self.gain = gain
        self.contact_bonus_on_wrapper = contact_bonus_on_wrapper # If you want wrapper to add its own bonus
        self.last_distance = None
        # Try to access dg from the underlying env if it exists (for contact bonus)
        self.contact_radius = getattr(self.unwrapped, 'dg', 2.0)


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action) # Get base reward from env
        
        # Agent position: obs[1] (x_sim_lon), obs[0] (y_sim_lat), obs[2] (z_alt)
        # Goal/Boom position: obs[12:15] (X, Y, Z world)
        agent_pos_sim_frame = obs[[1,0,2]] 
        goal_pos_world_frame = obs[12:15] 
        
        current_displacement_vec = goal_pos_world_frame - agent_pos_sim_frame
        distance = np.linalg.norm(current_displacement_vec)
        
        if self.last_distance is not None:
            # Shaping reward for getting closer
            reward += self.gain * (self.last_distance - distance)
        self.last_distance = distance

        # Optional: Add a bonus from the wrapper itself if in contact radius
        if self.contact_bonus_on_wrapper > 0 and distance < self.contact_radius:
             reward += self.contact_bonus_on_wrapper
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        agent_pos_sim_frame = obs[[1,0,2]]
        goal_pos_world_frame = obs[12:15]
        displacement = goal_pos_world_frame - agent_pos_sim_frame
        self.last_distance = np.linalg.norm(displacement)
        return obs, info


# Factory function for the original navigation environment
def wrap_jsbsim_navigate(**kwargs):
    env = JSBSimEnv(**kwargs)
    # Adjust gain for navigation; might need to be smaller if base rewards are large
    return PositionReward(env, gain=1e-2) 

# Factory function for the tanker environment
def wrap_jsbsim_tanker(**kwargs):
    env = JSBSimTankerEnv(**kwargs)
    # Gain for tanker approach; might need to be larger to encourage approach initially
    # The base tanker env already gives +50 for contact.
    return PositionReward(env, gain=5e-2, contact_bonus_on_wrapper=0.0) 

# Register the environments
gym.register(
    id="JSBSim-v0", # Original navigation
    entry_point=wrap_jsbsim_navigate,
    max_episode_steps=1200 # Approx 40 seconds at 30Hz env steps
)

gym.register(
    id="JSBSimTank-v0", # Tanker following
    entry_point=wrap_jsbsim_tanker,
    max_episode_steps=2400 # Approx 80 seconds, more time for rendezvous and station keeping
)


# Example script to test environments
if __name__ == "__main__":
    from time import sleep

    # Select which environment to test
    # test_env_id = "JSBSim-v0"
    test_env_id = "JSBSimTank-v0"

    print(f"Testing environment: {test_env_id}")
    # Create the unwrapped environment first for direct access if needed for debugging
    if test_env_id == "JSBSim-v0":
        raw_env = JSBSimEnv()
    else:
        raw_env = JSBSimTankerEnv()
    
    # Then wrap it if you want to test the wrapper too
    # env = PositionReward(raw_env, gain=1e-2 if test_env_id == "JSBSim-v0" else 5e-2)
    # Or use the registered factory
    env = gym.make(test_env_id)

    try:
        obs, info = env.reset()
        env.render()
        total_reward = 0
        for i in range(600): # Run for 20 sim seconds
            # Example action: slight pitch down, gentle roll, bit of throttle
            if test_env_id == "JSBSim-v0":
                action = np.array([0.0, -0.05, 0.0, 0.6], dtype=np.float32)
            else: # For tanker, try to maintain course/speed initially
                action = np.array([0.0, 0.0, 0.0, 0.55], dtype=np.float32) # Adjust throttle to match speed

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            
            if i % 30 == 0: # Print status every second
                if isinstance(env.unwrapped, JSBSimTankerEnv):
                    dist_to_boom = np.linalg.norm(env.unwrapped.boom_world_pos - env.unwrapped.agent_state[[1,0,2]])
                    print(f"Step: {i}, Action: {action}, Reward: {reward:.3f}, Total Reward: {total_reward:.3f}, Term: {terminated}, Trunc: {truncated}, DistBoom: {dist_to_boom:.2f}")
                else:
                    dist_to_goal = np.linalg.norm(env.unwrapped.goal_world_pos - env.unwrapped.agent_state[[1,0,2]])
                    print(f"Step: {i}, Action: {action}, Reward: {reward:.3f}, Total Reward: {total_reward:.3f}, Term: {terminated}, Trunc: {truncated}, DistGoal: {dist_to_goal:.2f}")


            if terminated or truncated:
                print(f"Episode finished after {i+1} steps. Final reward: {total_reward}")
                obs, info = env.reset()
                env.render()
                total_reward = 0
            
            sleep(1/env.metadata['render_fps']) # Match render FPS

    except KeyboardInterrupt:
        print("Test interrupted by user.")
    finally:
        env.close()
        print("Environment closed.")