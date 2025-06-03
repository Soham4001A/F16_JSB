import jsbsim
import gymnasium
import numpy as np
import collections
import math
import os

# Visualization imports
from .visualization.rendering import Viewer, load_mesh, load_shader, RenderObject, Grid, LineSegment
from .visualization.quaternion import Quaternion

# --- Constants ---
EARTH_RADIUS_METERS = 6.3781e6
FEET_TO_METERS = 0.3048
METERS_TO_FEET = 1.0 / FEET_TO_METERS
KNOTS_TO_MPS = 0.514444
NM_TO_METERS = 1852.0
METERS_TO_NM = 1.0 / NM_TO_METERS
RADIANS_TO_DEGREES = 180.0 / np.pi
DEGREES_TO_RADIANS = np.pi / 180.0
NUM_STACKED_FRAMES = 4
EPSILON = 1e-5

# JSBSim properties for the 737 ILS approach
ILS_STATE_PROPERTIES_737 = [
    "position/lat-geod-rad", "position/long-gc-deg",  # Use consistent geodetic
    "position/h-sl-ft", "position/h-agl-ft",
    "attitude/phi-rad", "attitude/theta-rad", "attitude/psi-rad",
    "velocities/vias-kts", "velocities/h-dot-fps", "aero/alpha-deg",
    "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec",
    "gear/unit[0]/wow", "gear/unit[1]/wow", "gear/unit[2]/wow", # Nose, Left, Right
    "propulsion/engine[0]/n1", "propulsion/engine[1]/n1",
    "propulsion/engine[0]/n2", "propulsion/engine[1]/n2",
    "propulsion/engine[0]/thrust-lbs", "propulsion/engine[1]/thrust-lbs",
]

# Indices for key properties
IDX_LAT_RAD = ILS_STATE_PROPERTIES_737.index("position/lat-geod-rad")
IDX_LON_RAD = ILS_STATE_PROPERTIES_737.index("position/long-gc-deg")
IDX_ALT_MSL_FT = ILS_STATE_PROPERTIES_737.index("position/h-sl-ft")
IDX_ALT_AGL_FT = ILS_STATE_PROPERTIES_737.index("position/h-agl-ft")
IDX_ROLL_RAD = ILS_STATE_PROPERTIES_737.index("attitude/phi-rad")
IDX_PITCH_RAD = ILS_STATE_PROPERTIES_737.index("attitude/theta-rad")
IDX_HEADING_RAD = ILS_STATE_PROPERTIES_737.index("attitude/psi-rad")
IDX_VIAS_KTS = ILS_STATE_PROPERTIES_737.index("velocities/vias-kts")
IDX_VS_FPS = ILS_STATE_PROPERTIES_737.index("velocities/h-dot-fps")
IDX_AOA_DEG = ILS_STATE_PROPERTIES_737.index("aero/alpha-deg")
IDX_ROLL_RATE_RAD_S = ILS_STATE_PROPERTIES_737.index("velocities/p-rad_sec")
IDX_PITCH_RATE_RAD_S = ILS_STATE_PROPERTIES_737.index("velocities/q-rad_sec")
IDX_YAW_RATE_RAD_S = ILS_STATE_PROPERTIES_737.index("velocities/r-rad_sec")
IDX_WOW_NOSE = ILS_STATE_PROPERTIES_737.index("gear/unit[0]/wow")
IDX_WOW_LEFT_MAIN = ILS_STATE_PROPERTIES_737.index("gear/unit[1]/wow")
IDX_WOW_RIGHT_MAIN = ILS_STATE_PROPERTIES_737.index("gear/unit[2]/wow")

OBS_FEATURE_NAMES = [
    "delta_localizer_deg", "delta_glideslope_deg", "airspeed_error_kts", "vertical_speed_fps",
    "pitch_angle_rad", "roll_angle_rad", "heading_error_rad", "altitude_agl_ft", "alpha_deg",
    "pitch_rate_rad_s", "roll_rate_rad_s", "distance_to_threshold_nm",
]
NUM_OBS_FEATURES = len(OBS_FEATURE_NAMES)

# TODO_737: WIDEN these bounds significantly for delta_loc, delta_gs, distance_to_threshold
# if starting from a JSBSim default location that could be far from the runway.
OBS_LOW = np.array([
    -10.0, -5.0, -30.0, -50.0,  # Reasonable ILS deviations for approach
    (-15.0 * DEGREES_TO_RADIANS), (-30.0 * DEGREES_TO_RADIANS),
    -np.pi, 0.0, -5.0,
    (-np.pi/6), (-np.pi/6), 0.0,
], dtype=np.float32)

OBS_HIGH = np.array([
    10.0, 5.0, 30.0, 20.0,      # Reasonable ILS deviations for approach  
    (20.0 * DEGREES_TO_RADIANS), (30.0 * DEGREES_TO_RADIANS),
    np.pi, 5000.0, 15.0,
    (np.pi/6), (np.pi/6), 20.0,  # Max 20 NM distance
], dtype=np.float32)


def normalize_angle_mpi_pi(angle_rad: float) -> float:
    if np.isnan(angle_rad) or np.isinf(angle_rad): return 0.0
    angle_rad = angle_rad % (2 * np.pi)
    if angle_rad >= np.pi: angle_rad -= 2 * np.pi
    return angle_rad

def normalize_angle_0_2pi(angle_rad: float) -> float:
    if np.isnan(angle_rad) or np.isinf(angle_rad): return 0.0
    return angle_rad % (2 * np.pi)

def geodetic_to_enu(lat_rad, lon_rad, alt_m, ref_lat_rad, ref_lon_rad, ref_alt_m):
    dx = EARTH_RADIUS_METERS * (lon_rad - ref_lon_rad) * math.cos(ref_lat_rad)
    dy = EARTH_RADIUS_METERS * (lat_rad - ref_lat_rad)
    dz = alt_m - ref_alt_m
    return dx, dy, dz

class Boeing737ILSEnv(gymnasium.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self,
                 jsbsim_root: str = '.', aircraft_model: str = '737', dt_hz: int = 10,
                 runway_lat_deg: float = 34.0, runway_lon_deg: float = -118.0, runway_alt_ft: float = 2300.0,
                 runway_hdg_deg: float = 220.0, glideslope_deg: float = 3.0,
                 target_approach_ias_kts: float = 135.0, flare_start_agl_ft: float = 50.0,
                 render_mode: str | None = None,
                 max_episode_steps: int = 2500): # Increased further for navigation
        super().__init__()
        
        self.render_mode = render_mode
        self.jsbsim_root = jsbsim_root
        self.aircraft_model = aircraft_model
        self.aircraft_render_obj = None
        self.glideslope_render_obj = None
        self._render_failed = False 
        self.dt_secs = 1.0 / dt_hz
        self.down_sample = max(1, int(round(120 / dt_hz)))

        self.runway_lat_rad = runway_lat_deg * DEGREES_TO_RADIANS
        self.runway_lon_rad = runway_lon_deg * DEGREES_TO_RADIANS
        self.runway_alt_m = runway_alt_ft * FEET_TO_METERS
        self.runway_hdg_rad = normalize_angle_0_2pi(runway_hdg_deg * DEGREES_TO_RADIANS)
        self.glideslope_rad = glideslope_deg * DEGREES_TO_RADIANS
        self.target_approach_ias_kts = target_approach_ias_kts
        self.flare_start_agl_ft = flare_start_agl_ft
        self.num_stacked_frames = NUM_STACKED_FRAMES

        self.obs_buffer = collections.deque(maxlen=self.num_stacked_frames)
        self.observation_space = gymnasium.spaces.Box(
            low=np.tile(OBS_LOW, (self.num_stacked_frames, 1)),
            high=np.tile(OBS_HIGH, (self.num_stacked_frames, 1)),
            shape=(self.num_stacked_frames, NUM_OBS_FEATURES), dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-1, -1, -1, 0.20], dtype=np.float32), # Min throttle at idle
            high=np.array([1, 1, 1, 1.0], dtype=np.float32),
            shape=(4,), dtype=np.float32)

        self.simulation = jsbsim.FGFDMExec(self.jsbsim_root, None)
        self.simulation.set_debug_level(0)
        print(f"[INIT] Loading model: {self.aircraft_model}")
        if not self.simulation.load_model(self.aircraft_model):
            raise RuntimeError(f"Failed to load JSBSim model: {self.aircraft_model}")
        print(f"[INIT] Model {self.aircraft_model} loaded successfully.")
        self.simulation.set_dt(self.dt_secs / self.down_sample)
        
        # TODO_737: VERIFY THESE FROM YOUR CFM56.xml
        self.CFM56_IDLE_N2_PERCENT = 58.5 # Example: from <idlen2>
        self.CFM56_APPROACH_N1_TARGET = 65.0 # Example: typical approach N1 target %

        self.current_step_in_episode = 0
        self.max_episode_steps = max_episode_steps
        self._reset_landing_state_vars()
        self.viewer = None
        self._render_failed = False  # Track if render init failed
        self.last_action = np.zeros(self.action_space.shape)
        self.prev_dist_to_thresh_nm = 0.0

    def _reset_landing_state_vars(self):
        self.main_gear_contact=False; self.nose_gear_contact=False; self.all_gear_contact=False
        self.main_gear_touchdown_time=-1.0; self.nose_gear_touchdown_time=-1.0
        self.pitch_at_main_touchdown_rad=0.0; self.vs_at_main_touchdown_fps=0.0
        self.roll_at_main_touchdown_rad=0.0; self.loc_dev_at_main_touchdown_deg=0.0
        self.is_flare_active=False; self.successfully_landed=False

    def _set_initial_conditions(self, rng):
        """Fixed initial conditions that properly position aircraft for ILS approach"""
        
        # 1. FIX: Properly set initial position relative to runway
        # Calculate a reasonable starting position for ILS intercept
        approach_distance_nm = rng.uniform(8, 15)  # 8-15 NM from runway
        approach_distance_m = approach_distance_nm * NM_TO_METERS
        
        # Position aircraft on extended centerline or offset for intercept
        intercept_angle_deg = rng.uniform(-30, 30)  # Intercept angle
        intercept_angle_rad = intercept_angle_deg * DEGREES_TO_RADIANS
        
        # Calculate initial lat/lon relative to runway
        # Back up along the runway centerline, then offset laterally
        runway_back_bearing_rad = self.runway_hdg_rad + np.pi
        
        # Initial position along extended centerline
        delta_east_m = approach_distance_m * math.sin(runway_back_bearing_rad)
        delta_north_m = approach_distance_m * math.cos(runway_back_bearing_rad)
        
        # Add lateral offset for intercept scenario
        lateral_offset_m = rng.uniform(-2000, 2000)  # ±2km lateral offset
        delta_east_m += lateral_offset_m * math.cos(runway_back_bearing_rad)
        delta_north_m += -lateral_offset_m * math.sin(runway_back_bearing_rad)
        
        # Convert to lat/lon
        delta_lat_rad = delta_north_m / EARTH_RADIUS_METERS
        delta_lon_rad = delta_east_m / (EARTH_RADIUS_METERS * math.cos(self.runway_lat_rad))
        
        initial_lat_deg = (self.runway_lat_rad + delta_lat_rad) * RADIANS_TO_DEGREES
        initial_lon_deg = (self.runway_lon_rad + delta_lon_rad) * RADIANS_TO_DEGREES
        
        # 2. FIX: Set proper initial conditions
        target_alt_msl_ft = self.runway_alt_m * METERS_TO_FEET + rng.uniform(2000, 4000)
        target_ias_kts = self.target_approach_ias_kts + rng.uniform(10, 30)
        
        # Initial heading for intercept
        if abs(intercept_angle_deg) < 5:  # Nearly aligned
            target_hdg_deg = self.runway_hdg_rad * RADIANS_TO_DEGREES + rng.uniform(-10, 10)
        else:  # Intercept scenario
            target_hdg_deg = (self.runway_hdg_rad * RADIANS_TO_DEGREES + intercept_angle_deg) % 360
        
        # 3. FIX: Actually set the lat/lon in JSBSim
        self.simulation.set_property_value('ic/lat-geod-deg', initial_lat_deg)
        self.simulation.set_property_value('ic/long-gc-deg', initial_lon_deg)
        self.simulation.set_property_value('ic/h-sl-ft', target_alt_msl_ft)
        self.simulation.set_property_value('ic/vc-kts', target_ias_kts)
        self.simulation.set_property_value('ic/psi-true-deg', target_hdg_deg)
        
        # Set reasonable flight path angle
        target_gamma_deg = rng.uniform(-3.0, -0.5)  # Slight descent
        target_alpha_deg = rng.uniform(2.0, 4.0)    # Reasonable AoA
        target_theta_deg = target_alpha_deg + target_gamma_deg
        
        self.simulation.set_property_value('ic/gamma-deg', target_gamma_deg)
        self.simulation.set_property_value('ic/alpha-deg', target_alpha_deg)
        self.simulation.set_property_value('ic/theta-deg', target_theta_deg)
        self.simulation.set_property_value('ic/phi-deg', rng.uniform(-3, 3))
        self.simulation.set_property_value('ic/beta-deg', 0.0)
        
        # 4. FIX: Proper engine initialization
        # Don't try to set N1/N2 directly - set throttle and let engine model respond
        approach_throttle = rng.uniform(0.4, 0.6)  # Reasonable approach power
        self.simulation.set_property_value("fcs/throttle-cmd-norm[0]", approach_throttle)
        self.simulation.set_property_value("fcs/throttle-cmd-norm[1]", approach_throttle)
        
        # Start engines properly
        self.simulation.set_property_value('propulsion/engine[0]/set_running', 1)
        self.simulation.set_property_value('propulsion/engine[1]/set_running', 1)
        
        # 5. FIX: Proper configuration for approach
        self.simulation.set_property_value("fcs/flap-cmd-norm", 0.25)  # Approach flaps
        self.simulation.set_property_value("gear/gear-cmd-norm", 1.0)  # Gear down for approach
        self.simulation.set_property_value("fcs/speedbrake-cmd-norm", 0.0)
        self.simulation.set_property_value("fcs/brake-parking-cmd-norm", 0.0)
        
        # Initialize and trim
        self.simulation.set_property_value("simulation/init_trim", 1)
        if not self.simulation.run_ic():
            print("[ERROR] JSBSim run_ic() failed!")
            return False
        self.simulation.set_property_value("simulation/init_trim", 0)
        
        # Stabilization runs
        for _ in range(50):
            self.simulation.run()
        
        return True

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        rng = self.np_random
        self._render_failed = False # Reset render failure flag for new episode
        self.current_step_in_episode = 0
        self._reset_landing_state_vars()
        self._set_initial_conditions(rng) # Call the method that uses JSBSim's default lat/lon

        self.obs_buffer.clear()
        raw_state_init = self._get_raw_jsbsim_state()
        initial_single_obs = self._get_observation(raw_state_init)
        
        # Initialize prev_dist_to_thresh_nm based on the actual starting position
        # This is important because the start distance will now be large
        self.prev_dist_to_thresh_nm = initial_single_obs[OBS_FEATURE_NAMES.index("distance_to_threshold_nm")] + 1.0 
        
        for _ in range(self.num_stacked_frames): self.obs_buffer.append(np.copy(initial_single_obs))
        initial_stacked_obs = np.array(self.obs_buffer, dtype=np.float32)
        
        if not self.observation_space.contains(initial_stacked_obs):
            print(f"ERROR IN RESET: Initial observation is NOT within defined space! First frame: {initial_single_obs}")
            # Add detailed print loop for feature bounds if needed
        return initial_stacked_obs, {"initial_state_debug": initial_single_obs.tolist()}

    def _get_raw_jsbsim_state(self) -> np.ndarray: # Ensure uses ILS_STATE_PROPERTIES_737
        state_values = np.zeros(len(ILS_STATE_PROPERTIES_737), dtype=np.float32)
        for i, prop_name in enumerate(ILS_STATE_PROPERTIES_737):
            try:
                val = self.simulation.get_property_value(prop_name)
                if val is None or np.isnan(val) or np.isinf(val): val = 0.0
                state_values[i] = float(val)
            except Exception: state_values[i] = 0.0
        return state_values

    def _calculate_ils_deviations_and_distances(self, raw_state: np.ndarray):
        ac_lat_rad = raw_state[IDX_LAT_RAD]
        ac_lon_deg = raw_state[IDX_LON_RAD]  # This is in degrees!
        ac_lon_rad = ac_lon_deg * DEGREES_TO_RADIANS  # Convert to radians
        ac_alt_msl_ft = raw_state[IDX_ALT_MSL_FT]
        ac_hdg_rad = normalize_angle_0_2pi(raw_state[IDX_HEADING_RAD])
        east_m, north_m, _ = geodetic_to_enu(
            ac_lat_rad, ac_lon_rad, ac_alt_msl_ft*FEET_TO_METERS,
            self.runway_lat_rad, self.runway_lon_rad, self.runway_alt_m
        )
        dist_to_thresh_m_2d = math.sqrt(east_m**2 + north_m**2)
        dist_to_thresh_nm = dist_to_thresh_m_2d * METERS_TO_NM
        delta_loc_deg, delta_gs_deg = 0.0, 0.0
        if dist_to_thresh_m_2d > 1.0: # Avoid division by zero if at threshold
            runway_perp_east = math.cos(self.runway_hdg_rad); runway_perp_north = -math.sin(self.runway_hdg_rad)
            lateral_dist_m = east_m * runway_perp_east + north_m * runway_perp_north
            delta_loc_rad = math.atan2(lateral_dist_m, dist_to_thresh_m_2d)
            delta_loc_deg = delta_loc_rad * RADIANS_TO_DEGREES
            ac_alt_agl_ft = raw_state[IDX_ALT_AGL_FT]
            desired_alt_agl_on_gs_m = dist_to_thresh_m_2d * math.tan(self.glideslope_rad)
            desired_alt_agl_on_gs_ft = desired_alt_agl_on_gs_m * METERS_TO_FEET
            alt_diff_ft = ac_alt_agl_ft - desired_alt_agl_on_gs_ft
            delta_gs_rad = math.atan2(alt_diff_ft * FEET_TO_METERS, dist_to_thresh_m_2d)
            delta_gs_deg = delta_gs_rad * RADIANS_TO_DEGREES
        heading_error_rad = normalize_angle_mpi_pi(ac_hdg_rad - self.runway_hdg_rad)
        return delta_loc_deg, delta_gs_deg, heading_error_rad, dist_to_thresh_nm

    def _get_observation(self, raw_state: np.ndarray):
        delta_loc, delta_gs, hdg_err, dist_nm = self._calculate_ils_deviations_and_distances(raw_state)
        #print(f"[OBS DEBUG] Delta Loc: {delta_loc:.2f} deg, Delta GS: {delta_gs:.2f} deg, Dist to Threshold: {dist_nm:.2f} NM")
        ias_err = raw_state[IDX_VIAS_KTS] - self.target_approach_ias_kts
        obs_features = np.array([
            delta_loc, delta_gs, ias_err, raw_state[IDX_VS_FPS],
            normalize_angle_mpi_pi(raw_state[IDX_PITCH_RAD]), normalize_angle_mpi_pi(raw_state[IDX_ROLL_RAD]),
            hdg_err, raw_state[IDX_ALT_AGL_FT], raw_state[IDX_AOA_DEG],
            raw_state[IDX_PITCH_RATE_RAD_S], raw_state[IDX_ROLL_RATE_RAD_S], dist_nm
        ], dtype=np.float32)
        # print(f"[OBS] {obs_features}")  # <-- Add this line to print the observation features
        assert len(obs_features) == NUM_OBS_FEATURES, "Obs feature count mismatch"
        obs_features_clipped = np.clip(obs_features, OBS_LOW, OBS_HIGH)
        return obs_features_clipped

    def _update_landing_gear_status(self, raw_state: np.ndarray):
        WOW_THRESHOLD = 0.1 # Ground contact if any force, not just > 0.5 for some models
        try:
            nose_on_ground = raw_state[IDX_WOW_NOSE] > WOW_THRESHOLD
            left_main_on_ground = raw_state[IDX_WOW_LEFT_MAIN] > WOW_THRESHOLD
            right_main_on_ground = raw_state[IDX_WOW_RIGHT_MAIN] > WOW_THRESHOLD
        except IndexError:
            print("[ERROR] WOW Property Index out of bounds. Check ILS_STATE_PROPERTIES_737 and IDX_WOW_* definitions.")
            nose_on_ground,left_main_on_ground,right_main_on_ground = False,False,False
        except Exception as e:
            print(f"[WARNING] Error accessing WOW properties: {e}")
            nose_on_ground,left_main_on_ground,right_main_on_ground = False,False,False

        current_time = self.current_step_in_episode * self.dt_secs
        if not self.main_gear_contact and left_main_on_ground and right_main_on_ground:
            self.main_gear_contact=True; self.main_gear_touchdown_time=current_time
            self.pitch_at_main_touchdown_rad=normalize_angle_mpi_pi(raw_state[IDX_PITCH_RAD])
            self.vs_at_main_touchdown_fps=raw_state[IDX_VS_FPS]
            self.roll_at_main_touchdown_rad=normalize_angle_mpi_pi(raw_state[IDX_ROLL_RAD])
            delta_loc,_,_,_ = self._calculate_ils_deviations_and_distances(raw_state)
            self.loc_dev_at_main_touchdown_deg=delta_loc
        if not self.nose_gear_contact and nose_on_ground:
            self.nose_gear_contact=True; self.nose_gear_touchdown_time=current_time
        if self.main_gear_contact and self.nose_gear_contact: self.all_gear_contact=True

    def _calculate_reward(self, raw_state, current_obs_features):
        """Fixed reward function with proper scaling"""
        reward = 0.0
        
        delta_loc_deg = current_obs_features[0]
        delta_gs_deg = current_obs_features[1] 
        airspeed_error_kts = current_obs_features[2]
        dist_to_thresh_nm = current_obs_features[11]
        
        # 8. FIX: Distance-based reward scaling
        if dist_to_thresh_nm > 10.0:
            # Far away - focus on navigation towards runway
            nav_reward = 0.1 * max(0, self.prev_dist_to_thresh_nm - dist_to_thresh_nm)
            reward += nav_reward
            
            # Gentle penalties when far
            if abs(delta_loc_deg) > 5.0:
                reward -= 0.01 * (abs(delta_loc_deg) - 5.0)
            if abs(airspeed_error_kts) > 20.0:
                reward -= 0.01 * (abs(airspeed_error_kts) - 20.0)
                
        elif dist_to_thresh_nm > 3.0:
            # Medium distance - focus on ILS capture
            if abs(delta_loc_deg) < 2.0:
                reward += 0.1  # Reward for being on localizer
            if abs(delta_gs_deg) < 1.0:
                reward += 0.1  # Reward for being on glideslope
                
            # Progressive penalties
            reward -= 0.02 * abs(delta_loc_deg)**1.5
            reward -= 0.02 * abs(delta_gs_deg)**1.5
            
        else:
            # Close to runway - precision approach
            reward -= 0.05 * abs(delta_loc_deg)**2
            reward -= 0.05 * abs(delta_gs_deg)**2
            
            # Airspeed management
            if abs(airspeed_error_kts) < 5.0:
                reward += 0.05
            else:
                reward -= 0.02 * abs(airspeed_error_kts)
        
        # Always update distance tracking
        self.prev_dist_to_thresh_nm = dist_to_thresh_nm
        
        # Attitude penalties (always applied)
        roll_rad = current_obs_features[5]
        pitch_rad = current_obs_features[4]
        
        if abs(roll_rad) > 30 * DEGREES_TO_RADIANS:
            reward -= 0.1 * (abs(roll_rad) - 30 * DEGREES_TO_RADIANS)**2
        if abs(pitch_rad) > 20 * DEGREES_TO_RADIANS:
            reward -= 0.1 * (abs(pitch_rad) - 20 * DEGREES_TO_RADIANS)**2
        
        return reward

    def _check_termination_truncation(self, raw_state: np.ndarray, current_obs_features: np.ndarray):
        terminated, success, failure_reason = False, False, "in_progress"
        alt_agl_ft=current_obs_features[7]; dist_nm=current_obs_features[11]
        delta_loc=current_obs_features[0]; delta_gs=current_obs_features[1]
        ias_err=current_obs_features[2]; roll_rad=current_obs_features[5]
        pitch_rad=current_obs_features[4]; aoa=current_obs_features[8]

        if alt_agl_ft < -5.0: terminated=True; failure_reason="crashed_alt"
        if raw_state[IDX_VS_FPS] < -25 and alt_agl_ft < 20: terminated=True; failure_reason="crashed_vs"
        if abs(roll_rad) > (45*DEGREES_TO_RADIANS): terminated=True; failure_reason="extreme_roll"
        if abs(pitch_rad) > (25*DEGREES_TO_RADIANS): terminated=True; failure_reason="extreme_pitch"
        
        STALL_AOA_737 = 17.0 
        if aoa > STALL_AOA_737 and raw_state[IDX_VIAS_KTS] < (self.target_approach_ias_kts - 40): terminated=True; failure_reason="stalled_737"

        # More generous out of bounds when far away
        if dist_nm > 15.0:
            if abs(delta_loc) > 60.0 : terminated=True; failure_reason="lost_course_far"
            if abs(ias_err) > 60.0 : terminated=True; failure_reason="extreme_airspeed_far"
        elif not self.is_flare_active and dist_nm > 0.1: # Stricter when closer
            if abs(delta_loc)>20.0: terminated=True; failure_reason="off_loc_near" # Was 12
            if abs(delta_gs)>10.0: terminated=True; failure_reason="off_gs_near"   # Was 6
            if abs(ias_err)>35.0: terminated=True; failure_reason="bad_airspeed_near"

        if dist_nm < -0.3 and not self.all_gear_contact: terminated=True; failure_reason="flew_past_runway"

        if self.main_gear_contact:
            MAX_SINK_737 = -10.0; MIN_PITCH_TD_737, MAX_PITCH_TD_737 = 0.0, 7.0 
            if self.vs_at_main_touchdown_fps < MAX_SINK_737: terminated=True; failure_reason="hard_landing_737"
            pitch_td_deg = self.pitch_at_main_touchdown_rad*RADIANS_TO_DEGREES
            if not (MIN_PITCH_TD_737 <= pitch_td_deg <= MAX_PITCH_TD_737): terminated=True; failure_reason="bad_pitch_td_737"
            if abs(self.roll_at_main_touchdown_rad) > (6*DEGREES_TO_RADIANS): terminated=True; failure_reason="roll_at_td_737"
            if abs(self.loc_dev_at_main_touchdown_deg)>3.0: terminated=True; failure_reason="off_center_td_737" # Wider for 737
        
        if self.nose_gear_contact and (not self.main_gear_contact or self.nose_gear_touchdown_time < (self.main_gear_touchdown_time-self.dt_secs*0.1)):
            terminated=True; failure_reason="nose_first_td_737"

        if self.all_gear_contact and not terminated: # Check if previous conditions already terminated
            if self.main_gear_touchdown_time < self.nose_gear_touchdown_time and \
               self.vs_at_main_touchdown_fps >= MAX_SINK_737 and \
               MIN_PITCH_TD_737 <= (self.pitch_at_main_touchdown_rad*RADIANS_TO_DEGREES) <= MAX_PITCH_TD_737 and \
               abs(self.roll_at_main_touchdown_rad) <= (5*DEGREES_TO_RADIANS) and \
               abs(self.loc_dev_at_main_touchdown_deg) <= 2.5: # Wider for success
                terminated=True; success=True; self.successfully_landed=True; failure_reason="success_737"
            elif not success: # If all gear down but conditions not met, still a bad landing
                terminated=True; failure_reason="bad_landing_all_gear_down"
        
        truncated = False
        if not terminated and self.current_step_in_episode >= self.max_episode_steps:
            truncated=True; failure_reason="truncated_max_steps"
        
        info={'success':success, 'failure_reason': failure_reason}
        return terminated, truncated, info

    def step(self, action: np.ndarray):
        self.current_step_in_episode += 1; self.last_action = action.copy()
        self.simulation.set_property_value("fcs/aileron-cmd-norm", float(action[0]))
        self.simulation.set_property_value("fcs/elevator-cmd-norm", float(action[1]))
        self.simulation.set_property_value("fcs/rudder-cmd-norm", float(action[2]))
        self.simulation.set_property_value("fcs/throttle-cmd-norm[0]", float(action[3]))
        self.simulation.set_property_value("fcs/throttle-cmd-norm[1]", float(action[3]))

        for _ in range(self.down_sample): self.simulation.run()
        raw_state = self._get_raw_jsbsim_state()
        self._update_landing_gear_status(raw_state)
        current_single_obs = self._get_observation(raw_state)
        self.obs_buffer.append(current_single_obs)
        stacked_obs = np.array(self.obs_buffer, dtype=np.float32)
        reward = self._calculate_reward(raw_state, current_single_obs)
        terminated, truncated, info = self._check_termination_truncation(raw_state, current_single_obs)

        if terminated: reward += 1000.0 if info.get('success',False) else -500.0 # Larger success/failure rewards
        elif truncated: reward -= 200.0
        
        if not self.observation_space.contains(stacked_obs): # Keep this for debugging
            print(f"ERROR IN STEP ({self.current_step_in_episode}): Obs NOT in space! Latest Frame: {current_single_obs}")
        return stacked_obs, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.current_step_in_episode = 0
        self._reset_landing_state_vars()
        self._set_initial_conditions(rng) 
        self.obs_buffer.clear()
        raw_state_init = self._get_raw_jsbsim_state()
        initial_single_obs = self._get_observation(raw_state_init)
        self.prev_dist_to_thresh_nm = initial_single_obs[OBS_FEATURE_NAMES.index("distance_to_threshold_nm")] + 1.0 # Add offset
        for _ in range(self.num_stacked_frames): self.obs_buffer.append(np.copy(initial_single_obs))
        initial_stacked_obs = np.array(self.obs_buffer, dtype=np.float32)
        
        if not self.observation_space.contains(initial_stacked_obs): # Keep for debugging
            print(f"ERROR IN RESET: Initial obs NOT in space! Initial Single Frame: {initial_single_obs}")
        return initial_stacked_obs, {"initial_state_debug": initial_single_obs.tolist()}

    def render(self):
        scale = 1e-5 # Your working scale factor

        effective_mode = self.render_mode
        headless_flag = False

        if effective_mode == 'human':
            if "DISPLAY" not in os.environ and not hasattr(os, 'geteuid'): # Basic check
                return None
            headless_flag = False
        elif effective_mode == 'rgb_array':
            headless_flag = True
        else:
            return None # No valid render mode

        if self._render_failed:
            return None

        if self.viewer is None:
            try:
                self.viewer = Viewer(1280, 720, headless=headless_flag)
                
                visualization_dir = os.path.join(os.path.dirname(__file__), "visualization")
                aircraft_mesh_filename = "737.obj" # Or your F-16 mesh
                
                mesh_path_aircraft = os.path.join(visualization_dir, aircraft_mesh_filename)
                if not os.path.exists(mesh_path_aircraft):
                     mesh_path_aircraft = os.path.join(os.path.dirname(__file__), aircraft_mesh_filename)
                if not os.path.exists(mesh_path_aircraft):
                    if os.path.exists(aircraft_mesh_filename): # Check CWD as last resort
                        mesh_path_aircraft = aircraft_mesh_filename
                    else:
                        print(f"[RENDER ERROR] CRITICAL: Aircraft mesh '{aircraft_mesh_filename}' not found.")
                        self._render_failed = True; self.close(); return None
                
                if self.viewer.ctx is None or self.viewer.prog is None:
                    raise RuntimeError("Viewer context or default program not initialized before mesh loading.")

                mesh_vao_aircraft = load_mesh(self.viewer.ctx, self.viewer.prog, mesh_path_aircraft)
                self.aircraft_render_obj = RenderObject(mesh_vao_aircraft)
                self.aircraft_render_obj.transform.scale = 1.0 / 3.0 # Your working aircraft model scale
                self.aircraft_render_obj.color = (0.1, 0.2, 0.5)
                self.viewer.objects.append(self.aircraft_render_obj)
                
                if self.viewer.unlit is None:
                     raise RuntimeError("Viewer unlit program not initialized before Grid/Line creation.")
                self.viewer.objects.append(Grid(self.viewer.ctx, self.viewer.unlit, 41, 5.0))

                # Initialize Glideslope Line Object
                dummy_p_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                dummy_p_b = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                self.glideslope_render_obj = LineSegment(self.viewer.ctx, self.viewer.unlit, dummy_p_a, dummy_p_b)
                self.glideslope_render_obj.color = (1.0, 0.8, 0.0) # Orange/Yellow
                self.viewer.objects.append(self.glideslope_render_obj)

            except Exception as e:
                print(f"[RENDER ERROR] Viewer or asset initialization error: {e}")
                import traceback; traceback.print_exc()
                self._render_failed = True
                self.close()
                return None
        
        if self.viewer is None or self.viewer.ctx is None : # Double check after init attempt
            self._render_failed = True 
            return None

        # --- State Fetching and Aircraft Scene Update ---
        last_raw_state = self._get_raw_jsbsim_state() # Ensure this method and IDX_ constants are correct
        ac_lat_rad = last_raw_state[IDX_LAT_RAD]
        ac_lon_rad = last_raw_state[IDX_LON_RAD]
        ac_alt_msl_ft = last_raw_state[IDX_ALT_MSL_FT]
        
        ac_phi_rad = normalize_angle_mpi_pi(last_raw_state[IDX_ROLL_RAD])
        ac_theta_rad = normalize_angle_mpi_pi(last_raw_state[IDX_PITCH_RAD])
        ac_psi_rad = normalize_angle_0_2pi(last_raw_state[IDX_HEADING_RAD])

        e_m, n_m, u_m = geodetic_to_enu(
            ac_lat_rad, ac_lon_rad, ac_alt_msl_ft * FEET_TO_METERS,
            self.runway_lat_rad, self.runway_lon_rad, self.runway_alt_m
        )
        
        aircraft_viewer_pos = np.array([e_m * scale, u_m * scale, n_m * scale], dtype=np.float32)
        self.aircraft_render_obj.transform.position = aircraft_viewer_pos

        q_body_to_world_jsb_convention = Quaternion.from_euler(ac_phi_rad, ac_theta_rad, ac_psi_rad, mode='XYZ')
        self.aircraft_render_obj.transform.rotation = Quaternion(
            q_body_to_world_jsb_convention.w,
           -q_body_to_world_jsb_convention.y, 
           -q_body_to_world_jsb_convention.z, 
            q_body_to_world_jsb_convention.x
        )

        # --- Update Glideslope Line ---
        if self.glideslope_render_obj:
            gs_point_a_viewer = np.array([0.0, 0.0, 0.0], dtype=np.float32) 

            GLIDESLOPE_LINE_VIEWER_LENGTH_HORIZONTAL_PROJECTION = 50.0 
            
            gs_point_b_up_viewer = GLIDESLOPE_LINE_VIEWER_LENGTH_HORIZONTAL_PROJECTION * math.tan(self.glideslope_rad)
            approach_hdg_rad = normalize_angle_0_2pi(self.runway_hdg_rad + np.pi)

            gs_point_b_east_viewer = GLIDESLOPE_LINE_VIEWER_LENGTH_HORIZONTAL_PROJECTION * math.sin(approach_hdg_rad)
            gs_point_b_north_viewer = GLIDESLOPE_LINE_VIEWER_LENGTH_HORIZONTAL_PROJECTION * math.cos(approach_hdg_rad)
            
            gs_point_b_viewer = np.array([
                gs_point_b_east_viewer,
                gs_point_b_up_viewer,
                gs_point_b_north_viewer
            ], dtype=np.float32)

            self.glideslope_render_obj.update_points(gs_point_a_viewer, gs_point_b_viewer)

        # --- Camera Setup ---
        CAM_EYE_OFFSET_BEHIND_AIRCRAFT = 15.0 
        CAM_EYE_VERTICAL_OFFSET = 5.0      
        dir_behind_aircraft_viewer_world = np.array([
            -math.sin(ac_psi_rad), 0.0, -math.cos(ac_psi_rad)
        ], dtype=np.float32)
        cam_eye_position = (
            aircraft_viewer_pos + 
            dir_behind_aircraft_viewer_world * CAM_EYE_OFFSET_BEHIND_AIRCRAFT + 
            np.array([0, CAM_EYE_VERTICAL_OFFSET, 0], dtype=np.float32)
        )
        camera_target_on_aircraft = aircraft_viewer_pos + np.array([0, 1.0, 0], dtype=np.float32)
        look_direction_vec = camera_target_on_aircraft - cam_eye_position
        dist_look = np.linalg.norm(look_direction_vec)
        if dist_look > 1e-6: 
            look_direction_vec /= dist_look
        else: 
            look_direction_vec = -dir_behind_aircraft_viewer_world
        
        r_cam_x, r_cam_y, r_cam_z = look_direction_vec[0], look_direction_vec[1], look_direction_vec[2]
        cam_yaw_rad = np.arctan2(-r_cam_x, -r_cam_z)
        cam_pitch_rad = np.arctan2(-r_cam_y, math.sqrt(r_cam_x**2 + r_cam_z**2 + 1e-9))
        
        try:
            camera_rotation_q = Quaternion.from_euler(-cam_pitch_rad, cam_yaw_rad, 0, mode=1) 
            self.viewer.set_view(
                x=cam_eye_position[0], y=cam_eye_position[1], z=cam_eye_position[2],
                rotation_quaternion=camera_rotation_q
            )
            self.viewer.render() # Render all objects to the FBO

        except Exception as e:
            print(f"[RENDER ERROR] Error during view_set or viewer.render(): {e}")
            import traceback; traceback.print_exc()
            self._render_failed = True
            return None

        # --- Get Frame ---
        if effective_mode == 'rgb_array':
            try:
                frame = self.viewer.get_frame()
                if frame is None: # Should ideally not happen if viewer.render() and get_frame() are robust
                    return np.zeros((self.viewer.height, self.viewer.width, 3), dtype=np.uint8)
                return frame
            except Exception as e:
                print(f"[RENDER ERROR] Error during self.viewer.get_frame(): {e}")
                import traceback; traceback.print_exc()
                self._render_failed = True
                return None # Return None as per Gymnasium spec on error
        
        elif effective_mode == 'human':
            # For 'human' mode, viewer.render() (called above) updates the Pygame window.
            # Gymnasium spec is to return None.
            return None 
        
        return None # Fallback, should not be reached if effective_mode is valid

    def close(self):
        if self.viewer: self.viewer.close(); self.viewer = None

gymnasium.register(
    id="Boeing737ILSEnv-v0",
    entry_point="jsbsim_gym.ils_env:Boeing737ILSEnv", # MAKE SURE THIS PATH IS CORRECT
    max_episode_steps=2500, # Increased from 2000
)

if __name__ == "__main__":
    from time import sleep
    print("Creating and testing Boeing737ILSEnv environment...")
    try:
        # For direct instantiation:
        env = Boeing737ILSEnv(jsbsim_root='.', render_mode=None) # Pass jsbsim_root
        # Or using gym.make if your entry_point in registration is correct for your file structure:
        # env = gymnasium.make("Boeing737ILSEnv-v0", jsbsim_root='.')

        print(f"Obs Space: {env.observation_space.shape}, Act Space: {env.action_space.shape}")
        for ep in range(1): # Test one episode
            print(f"\n--- Episode {ep + 1} ---")
            obs, info = env.reset(seed=42 + ep)
            print(f"Initial single obs (for debug): {info.get('initial_state_debug')}")
            term, trunc, tot_rew, ep_steps = False, False, 0, 0
            
            # Action: small positive pitch, slight roll right, bit of rudder, moderate throttle
            # test_action = np.array([0.05, 0.02, 0.01, 0.55], dtype=np.float32)
            # Action: Try to fly straight and level initially
            test_action = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)


            for i in range(env.max_episode_steps + 50):
                action = test_action # Use a fixed action for this test
                # If testing responsiveness:
                # if ep_steps > 50 and ep_steps < 100: action[1] = 0.05 # Pitch up
                # elif ep_steps > 150 and ep_steps < 200: action[0] = 0.1 # Roll right
                
                obs, reward, term, trunc, info_step = env.step(action)
                tot_rew += reward; ep_steps += 1
                if ep_steps % 100 == 0 or term or trunc:
                    print(f"S {ep_steps}, R: {reward:.3f}, TR: {tot_rew:.3f}, Term: {term}, Trunc: {trunc}, Fail: {info_step.get('failure_reason')}")
                    print(f"  Obs: DL:{obs[-1][0]:.1f} DG:{obs[-1][1]:.1f} IAS_err:{obs[-1][2]:.1f} VS:{obs[-1][3]:.1f} Pit:{obs[-1][4]*RADIANS_TO_DEGREES:.1f} Rol:{obs[-1][5]*RADIANS_TO_DEGREES:.1f} Hdg_err:{obs[-1][6]*RADIANS_TO_DEGREES:.1f} AGL:{obs[-1][7]:.1f} AoA:{obs[-1][8]:.1f} PRt:{obs[-1][9]:.2f} RRt:{obs[-1][10]:.2f} Dist:{obs[-1][11]:.1f}")
                if term or trunc:
                    print(f"Ep finished: {ep_steps} steps. TotalR: {tot_rew:.2f}. Success: {info_step.get('success')}. Reason: {info_step.get('failure_reason')}")
                    break
            print(f"Final Info Ep {ep+1}: {info_step}")
    except Exception as e:
        import traceback
        print(f"Error during testing: {e}")
        traceback.print_exc()
    finally:
        if 'env' in locals() and hasattr(env, 'close'): env.close(); print("Env closed.")