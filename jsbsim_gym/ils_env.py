import jsbsim
import gymnasium
import numpy as np
import collections
import math
import os # Not strictly needed by this version, but good if writing temp IC files

# Visualization imports (ensure these paths are correct relative to this file)
from .visualization.rendering import Viewer, load_mesh, load_shader, RenderObject, Grid
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
    "position/lat-geod-rad", "position/lon-geoc-rad", "position/h-sl-ft", "position/h-agl-ft",
    "attitude/phi-rad", "attitude/theta-rad", "attitude/psi-rad",
    "velocities/vias-kts", "velocities/h-dot-fps", "aero/alpha-deg",
    "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec",
    "gear/unit[0]/wow", # Nose Gear WOW (assuming index 0 is Nose from 737.xml)
    "gear/unit[1]/wow", # Left Main Gear WOW (assuming index 1 is Left Main)
    "gear/unit[2]/wow", # Right Main Gear WOW (assuming index 2 is Right Main)
    "propulsion/engine[0]/n1", "propulsion/engine[1]/n1",
    "propulsion/engine[0]/n2", "propulsion/engine[1]/n2",
    "propulsion/engine[0]/thrust-lbs", "propulsion/engine[1]/thrust-lbs",
]

# Indices for key properties
IDX_LAT_RAD = ILS_STATE_PROPERTIES_737.index("position/lat-geod-rad")
IDX_LON_RAD = ILS_STATE_PROPERTIES_737.index("position/lon-geoc-rad")
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

# Adjusted bounds for 737
OBS_LOW = np.array([
    -45.0, -15.0, -40.0, -50.0, 
    (-15.0 * DEGREES_TO_RADIANS) - EPSILON, 
    (-30.0 * DEGREES_TO_RADIANS) - EPSILON, 
    -np.pi - EPSILON, -10.0, -2.0,        
    (-np.pi/6) - EPSILON, (-np.pi/6) - EPSILON, 0.0,         
], dtype=np.float32)

OBS_HIGH = np.array([
    45.0, 15.0, 40.0, 15.0,        
    (20.0 * DEGREES_TO_RADIANS) + EPSILON,  
    (30.0 * DEGREES_TO_RADIANS) + EPSILON,  
    np.pi + EPSILON, 15000.0, 18.0,       
    (np.pi/6) + EPSILON,  (np.pi/6) + EPSILON,  30.0,        
], dtype=np.float32)


def normalize_angle_mpi_pi(angle_rad: float) -> float:
    if np.isnan(angle_rad) or np.isinf(angle_rad): return 0.0
    angle_rad = angle_rad % (2 * np.pi);
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
                 max_episode_steps: int = 2000):
        super().__init__()

        self.jsbsim_root = jsbsim_root
        self.aircraft_model = aircraft_model
        self.dt_secs = 1.0 / dt_hz
        self.down_sample = max(1, int(round(120 / dt_hz))) # Default JSBSim rate is often 120Hz for some models

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
            low=np.array([-1, -1, -1, 0.0], dtype=np.float32),
            high=np.array([1, 1, 1, 1.0], dtype=np.float32),
            shape=(4,), dtype=np.float32)

        self.simulation = jsbsim.FGFDMExec(self.jsbsim_root, None)
        self.simulation.set_debug_level(0)
        print(f"[INIT] Loading model: {self.aircraft_model}")
        if not self.simulation.load_model(self.aircraft_model):
            raise RuntimeError(f"Failed to load JSBSim model: {self.aircraft_model}")
        print(f"[INIT] Model {self.aircraft_model} loaded successfully.")
        self.simulation.set_dt(self.dt_secs / self.down_sample)
        
        # TODO_737: VERIFY THIS FROM YOUR CFM56.xml's <idlen2> tag!
        self.CFM56_IDLE_N2_PERCENT = 58.5 # Example value, adjust based on CFM56.xml

        self.current_step_in_episode = 0
        self.max_episode_steps = max_episode_steps
        self._reset_landing_state_vars()
        self.viewer = None
        self.last_action = np.zeros(self.action_space.shape) # Initialize last_action
        self.prev_dist_to_thresh_nm = 0.0 # Initialized in reset

    def _reset_landing_state_vars(self):
        self.main_gear_contact=False; self.nose_gear_contact=False; self.all_gear_contact=False
        self.main_gear_touchdown_time=-1.0; self.nose_gear_touchdown_time=-1.0
        self.pitch_at_main_touchdown_rad=0.0; self.vs_at_main_touchdown_fps=0.0
        self.roll_at_main_touchdown_rad=0.0; self.loc_dev_at_main_touchdown_deg=0.0
        self.is_flare_active=False; self.successfully_landed=False

    def _set_initial_conditions_for_ils(self, rng):
        print("[IC DEBUG 737] Setting ICs...")

        self.simulation.set_property_value('propulsion/engine[0]/set_running', 1)
        self.simulation.set_property_value('propulsion/engine[1]/set_running', 1)
        self.simulation.set_property_value("propulsion/engine[0]/n2", self.CFM56_IDLE_N2_PERCENT)
        self.simulation.set_property_value("propulsion/engine[1]/n2", self.CFM56_IDLE_N2_PERCENT)
        
        IDLE_THROTTLE_CMD = 0.22 
        self.simulation.set_property_value("fcs/throttle-cmd-norm[0]", IDLE_THROTTLE_CMD)
        self.simulation.set_property_value("fcs/throttle-cmd-norm[1]", IDLE_THROTTLE_CMD)

        self.simulation.set_property_value("fcs/brake-parking-cmd-norm", 0.0)
        self.simulation.set_property_value("fcs/left-brake-cmd-norm", 0.0) 
        self.simulation.set_property_value("fcs/right-brake-cmd-norm", 0.0)

        start_dist_nm = rng.uniform(15.0, 25.0)
        start_dist_m = start_dist_nm * NM_TO_METERS
        lateral_offset_nm = rng.uniform(-1.5, 1.5)
        lateral_offset_m = lateral_offset_nm * NM_TO_METERS
        
        dx_centerline = start_dist_m * math.sin(self.runway_hdg_rad)
        dy_centerline = start_dist_m * math.cos(self.runway_hdg_rad)
        dx_offset = lateral_offset_m * math.cos(self.runway_hdg_rad)
        dy_offset = -lateral_offset_m * math.sin(self.runway_hdg_rad)
        
        target_lon_rad_gc = self.runway_lon_rad + (dx_centerline + dx_offset) / (EARTH_RADIUS_METERS * math.cos(self.runway_lat_rad))
        target_lat_rad_geod = self.runway_lat_rad + (dy_centerline + dy_offset) / EARTH_RADIUS_METERS
        
        height_on_gs_m = math.tan(self.glideslope_rad) * start_dist_m
        target_alt_msl_m = self.runway_alt_m + height_on_gs_m
        target_alt_msl_ft = (target_alt_msl_m * METERS_TO_FEET) - rng.uniform(300, 700)
        target_alt_msl_ft = max(target_alt_msl_ft, self.runway_alt_m * METERS_TO_FEET + 2500)

        target_ias_kts = self.target_approach_ias_kts + rng.uniform(-5, 10)
        target_vc_kts = target_ias_kts 
        target_hdg_deg = (self.runway_hdg_rad * RADIANS_TO_DEGREES) + rng.uniform(-30,30)
        target_hdg_deg = target_hdg_deg % 360.0
        target_gamma_deg = -self.glideslope_rad * RADIANS_TO_DEGREES 
        target_alpha_deg = 3.5 # Adjusted for 737 initial approach/descent
        target_theta_deg = target_alpha_deg + target_gamma_deg 

        print(f"[IC DEBUG 737] Attempting: lat={target_lat_rad_geod*RADIANS_TO_DEGREES:.4f}, lon_gc={target_lon_rad_gc*RADIANS_TO_DEGREES:.4f}, alt={target_alt_msl_ft:.1f}, vc={target_vc_kts:.1f}, gamma={target_gamma_deg:.1f}, alpha={target_alpha_deg:.1f}, theta={target_theta_deg:.1f}, hdg={target_hdg_deg:.1f}")

        self.simulation.set_property_value('ic/lat-geod-deg', target_lat_rad_geod * RADIANS_TO_DEGREES)
        self.simulation.set_property_value('ic/long-gc-deg', target_lon_rad_gc * RADIANS_TO_DEGREES) 
        self.simulation.set_property_value('ic/h-sl-ft', target_alt_msl_ft)
        self.simulation.set_property_value('ic/vc-kts', target_vc_kts)
        self.simulation.set_property_value('ic/gamma-deg', target_gamma_deg)
        self.simulation.set_property_value('ic/alpha-deg', target_alpha_deg) 
        self.simulation.set_property_value('ic/beta-deg', 0.0)
        self.simulation.set_property_value('ic/phi-deg', 0.0 + rng.uniform(-3,3))
        self.simulation.set_property_value('ic/theta-deg', target_theta_deg) 
        self.simulation.set_property_value('ic/psi-true-deg', target_hdg_deg)
        
        # TODO_737: Flap setting needs verification for 737 model (0.0-1.0 range to actual flaps)
        # Flaps 15 for 737 is approx 0.375 if 1.0 is Flaps 40. Flaps 25 is approx 0.625.
        approach_flap_setting = 0.625 # Example: Flaps 25
        self.simulation.set_property_value("fcs/flap-cmd-norm", approach_flap_setting)
        self.simulation.set_property_value("gear/gear-cmd-norm", 1.0)
        self.simulation.set_property_value("fcs/speedbrake-cmd-norm", 0.0)
        
        approach_throttle_cmd = 0.45 + rng.uniform(0.0, 0.1) 
        self.simulation.set_property_value("fcs/throttle-cmd-norm[0]", approach_throttle_cmd)
        self.simulation.set_property_value("fcs/throttle-cmd-norm[1]", approach_throttle_cmd)
        
        self.simulation.set_property_value("simulation/init_trim", 1) 

        print("[IC DEBUG 737] Running run_ic()...")
        if not self.simulation.run_ic(): print("[IC DEBUG ERROR 737] run_ic() failed.")
        
        self.simulation.set_property_value("simulation/init_trim", 0)

        print("[IC DEBUG 737] Stabilization after ICs (30 steps)...")
        for i in range(30):
            self.simulation.run()
            if i % 10 == 0: # Print less frequently
                try:
                    ias = self.simulation.get_property_value("velocities/vias-kts")
                    n1_0 = self.simulation.get_property_value("propulsion/engine[0]/n1")
                    n2_0 = self.simulation.get_property_value("propulsion/engine[0]/n2")
                    lon = self.simulation.get_property_value("position/long-gc-rad") * RADIANS_TO_DEGREES
                    alt = self.simulation.get_property_value("position/h-sl-ft")
                    print(f"[IC DEBUG 737] Stab {i}: IAS={ias:.1f} N1={n1_0:.1f} N2={n2_0:.1f} Lon={lon:.4f} Alt={alt:.1f}")
                except Exception as e: print(f"[IC DEBUG 737] Error in stab print: {e}")


        current_raw_state = self._get_raw_jsbsim_state()
        actual_lon_gc = current_raw_state[IDX_LON_RAD] * RADIANS_TO_DEGREES
        actual_ias = current_raw_state[IDX_VIAS_KTS]
        print(f"[IC DEBUG 737] Actual state: Lon_GC={actual_lon_gc:.4f}, IAS={actual_ias:.1f}")

    def _get_raw_jsbsim_state(self) -> np.ndarray:
        state_values = np.zeros(len(ILS_STATE_PROPERTIES_737), dtype=np.float32)
        for i, prop_name in enumerate(ILS_STATE_PROPERTIES_737):
            try:
                val = self.simulation.get_property_value(prop_name)
                if val is None or np.isnan(val) or np.isinf(val): val = 0.0
                state_values[i] = float(val)
            except Exception: state_values[i] = 0.0
        return state_values

    def _calculate_ils_deviations_and_distances(self, raw_state: np.ndarray):
        ac_lat_rad = raw_state[IDX_LAT_RAD]; ac_lon_rad = raw_state[IDX_LON_RAD]
        ac_alt_msl_ft = raw_state[IDX_ALT_MSL_FT]; ac_hdg_rad = normalize_angle_0_2pi(raw_state[IDX_HEADING_RAD])
        east_m, north_m, _ = geodetic_to_enu(ac_lat_rad, ac_lon_rad, ac_alt_msl_ft*FEET_TO_METERS, self.runway_lat_rad, self.runway_lon_rad, self.runway_alt_m)
        dist_to_thresh_m_2d = math.sqrt(east_m**2 + north_m**2)
        dist_to_thresh_nm = dist_to_thresh_m_2d * METERS_TO_NM
        delta_loc_deg, delta_gs_deg = 0.0, 0.0
        if dist_to_thresh_m_2d > 1.0:
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
        ias_err = raw_state[IDX_VIAS_KTS] - self.target_approach_ias_kts
        obs_features = np.array([
            delta_loc, delta_gs, ias_err, raw_state[IDX_VS_FPS],
            normalize_angle_mpi_pi(raw_state[IDX_PITCH_RAD]), normalize_angle_mpi_pi(raw_state[IDX_ROLL_RAD]),
            hdg_err, raw_state[IDX_ALT_AGL_FT], raw_state[IDX_AOA_DEG],
            raw_state[IDX_PITCH_RATE_RAD_S], raw_state[IDX_ROLL_RATE_RAD_S], dist_nm
        ], dtype=np.float32)
        assert len(obs_features) == NUM_OBS_FEATURES, "Obs feature count mismatch"
        # Clipping here as a safety net if ICs are still sometimes out of extreme bounds
        obs_features_clipped = np.clip(obs_features, OBS_LOW, OBS_HIGH)
        return obs_features_clipped


    def _update_landing_gear_status(self, raw_state: np.ndarray):
        WOW_THRESHOLD = 0.5 
        try:
            # Using indexed WOW properties as they are more standard in JSBSim unless named ones are confirmed
            nose_on_ground = raw_state[IDX_WOW_NOSE] > WOW_THRESHOLD
            left_main_on_ground = raw_state[IDX_WOW_LEFT_MAIN] > WOW_THRESHOLD
            right_main_on_ground = raw_state[IDX_WOW_RIGHT_MAIN] > WOW_THRESHOLD
        except IndexError: # If IDX_WOW_* are out of bounds for ILS_STATE_PROPERTIES_737
            print("[ERROR] WOW Property Index out of bounds. Check ILS_STATE_PROPERTIES_737 and IDX_WOW_* definitions.")
            nose_on_ground,left_main_on_ground,right_main_on_ground = False,False,False
        except Exception as e: # Catch other potential errors during property access
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

    def _calculate_reward(self, raw_state: np.ndarray, current_obs_features: np.ndarray):
        reward = 0.0
        delta_loc_deg=current_obs_features[0]; delta_gs_deg=current_obs_features[1]
        airspeed_error_kts=current_obs_features[2]; roll_angle_rad=current_obs_features[5]
        pitch_angle_rad=current_obs_features[4]; aoa_deg=current_obs_features[8]
        alt_agl_ft=current_obs_features[7]; dist_to_thresh_nm=current_obs_features[11]

        loc_tol,gs_tol,spd_tol = 0.7,0.3,7.0 
        if abs(delta_loc_deg) < loc_tol: reward += 0.1
        else: reward -= 0.05 * (abs(delta_loc_deg) - loc_tol)**2 # Reduced penalty factor
        if not self.is_flare_active:
            if abs(delta_gs_deg) < gs_tol: reward += 0.1
            else: reward -= 0.05 * (abs(delta_gs_deg) - gs_tol)**2
        if abs(airspeed_error_kts) < spd_tol: reward += 0.05
        else: reward -= 0.015 * (abs(airspeed_error_kts) - spd_tol)**2 # Reduced penalty factor
        
        reward -= 0.03 * (abs(roll_angle_rad)/(20*DEGREES_TO_RADIANS))**2 
        if alt_agl_ft > self.flare_start_agl_ft + 20:
             if abs(pitch_angle_rad) > (12*DEGREES_TO_RADIANS): 
                reward -= 0.05 * (abs(pitch_angle_rad)-(12*DEGREES_TO_RADIANS))**2
        
        max_aoa_approach_737 = 10.0 # Stricter for 737 steady approach
        if not self.is_flare_active and aoa_deg > max_aoa_approach_737 :
            reward -= 0.05 * (aoa_deg - max_aoa_approach_737)**2

        if alt_agl_ft <= self.flare_start_agl_ft and not self.main_gear_contact: self.is_flare_active = True
        if self.is_flare_active and not self.main_gear_contact:
            vs_fps = raw_state[IDX_VS_FPS]
            if -6.0 < vs_fps < -0.5: reward += 0.15 
            elif vs_fps >= -0.5: reward -= 0.15
            else: reward -= 0.04 * abs(vs_fps - (-6.0))**1.5 # Softer penalty for high sink in flare initially
            
            target_flare_pitch_deg_737 = 2.5 
            if abs(pitch_angle_rad*RADIANS_TO_DEGREES - target_flare_pitch_deg_737) < 1.5: reward += 0.15
            else: reward -= 0.04 * (abs(pitch_angle_rad*RADIANS_TO_DEGREES - target_flare_pitch_deg_737)-1.5)**2
        
        if alt_agl_ft > self.flare_start_agl_ft * 1.5 :
            if dist_to_thresh_nm < self.prev_dist_to_thresh_nm and dist_to_thresh_nm > 0.1: 
                 reward += 0.02 * (self.prev_dist_to_thresh_nm - dist_to_thresh_nm) 
        self.prev_dist_to_thresh_nm = dist_to_thresh_nm
        reward += 0.001 # Smaller keep alive
        return reward

    def _check_termination_truncation(self, raw_state: np.ndarray, current_obs_features: np.ndarray):
        terminated, success = False, False
        alt_agl_ft=current_obs_features[7]; dist_nm=current_obs_features[11]
        delta_loc=current_obs_features[0]; delta_gs=current_obs_features[1]
        ias_err=current_obs_features[2]; roll_rad=current_obs_features[5]
        pitch_rad=current_obs_features[4]; aoa=current_obs_features[8]

        if alt_agl_ft < -5.0: terminated=True; failure_reason="crashed_alt" # Allow slightly more negative for ground compression
        if raw_state[IDX_VS_FPS] < -25 and alt_agl_ft < 20: terminated=True; failure_reason="crashed_vs"
        if abs(roll_rad) > (40*DEGREES_TO_RADIANS): terminated=True; failure_reason="extreme_roll" # Stricter for 737
        if abs(pitch_rad) > (20*DEGREES_TO_RADIANS): terminated=True; failure_reason="extreme_pitch" # Stricter for 737
        
        STALL_AOA_737 = 17.0 # Approx stall AoA for 737 with flaps
        if aoa > STALL_AOA_737 and raw_state[IDX_VIAS_KTS] < (self.target_approach_ias_kts - 30): terminated=True; failure_reason="stalled_737"

        if not self.is_flare_active and dist_nm > 0.15: # Only check far out
            if abs(delta_loc)>12.0: terminated=True; failure_reason="off_loc_far"
            if abs(delta_gs)>6.0: terminated=True; failure_reason="off_gs_far"
            if abs(ias_err)>35.0: terminated=True; failure_reason="bad_airspeed_far"
        if dist_nm < -0.2 and not self.all_gear_contact: terminated=True; failure_reason="flew_past_runway" # Ran off end

        if self.main_gear_contact:
            MAX_SINK_737 = -10.0 # -600 fpm
            MIN_PITCH_TD_737, MAX_PITCH_TD_737 = 0.0, 6.0 # Adjusted for 737
            if self.vs_at_main_touchdown_fps < MAX_SINK_737: terminated=True; failure_reason="hard_landing_737"
            pitch_td_deg = self.pitch_at_main_touchdown_rad*RADIANS_TO_DEGREES
            if not (MIN_PITCH_TD_737 <= pitch_td_deg <= MAX_PITCH_TD_737): terminated=True; failure_reason="bad_pitch_td_737"
            if abs(self.roll_at_main_touchdown_rad) > (6*DEGREES_TO_RADIANS): terminated=True; failure_reason="roll_at_td_737"
            if abs(self.loc_dev_at_main_touchdown_deg)>2.5: terminated=True; failure_reason="off_center_td_737"
        
        if self.nose_gear_contact and (not self.main_gear_contact or self.nose_gear_touchdown_time < (self.main_gear_touchdown_time - self.dt_secs*0.2)):
            terminated=True; failure_reason="nose_first_td_737"

        if self.all_gear_contact and not terminated:
            if self.main_gear_touchdown_time < self.nose_gear_touchdown_time and \
               self.vs_at_main_touchdown_fps >= MAX_SINK_737 and \
               MIN_PITCH_TD_737 <= (self.pitch_at_main_touchdown_rad*RADIANS_TO_DEGREES) <= MAX_PITCH_TD_737 and \
               abs(self.roll_at_main_touchdown_rad) <= (5*DEGREES_TO_RADIANS) and \
               abs(self.loc_dev_at_main_touchdown_deg) <= 2.0: 
                terminated=True; success=True; self.successfully_landed=True; failure_reason="success_737"
        
        truncated = False
        if not terminated and self.current_step_in_episode >= self.max_episode_steps:
            truncated=True; failure_reason="truncated_max_steps"
        
        info={'success':success, 'failure_reason': failure_reason if (terminated or truncated) else "none"}
        return terminated, truncated, info

    def step(self, action: np.ndarray):
        self.current_step_in_episode += 1; self.last_action = action
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

        if terminated: reward += 500.0 if info.get('success',False) else -250.0 # Increased failure penalty
        elif truncated: reward -= 150.0 # Increased truncation penalty
        
        if not self.observation_space.contains(stacked_obs):
            print(f"ERROR IN STEP ({self.current_step_in_episode}): Obs NOT in space! Latest Frame: {current_single_obs}")
        return stacked_obs, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.current_step_in_episode = 0
        self._reset_landing_state_vars()
        self._set_initial_conditions_for_ils(rng)
        self.obs_buffer.clear()
        raw_state_init = self._get_raw_jsbsim_state()
        initial_single_obs = self._get_observation(raw_state_init)
        self.prev_dist_to_thresh_nm = initial_single_obs[OBS_FEATURE_NAMES.index("distance_to_threshold_nm")] + 1.0
        for _ in range(self.num_stacked_frames): self.obs_buffer.append(np.copy(initial_single_obs))
        initial_stacked_obs = np.array(self.obs_buffer, dtype=np.float32)
        
        if not self.observation_space.contains(initial_stacked_obs):
            print(f"ERROR IN RESET: Initial obs NOT in space! Initial Single Frame: {initial_single_obs}")
        return initial_stacked_obs, {"initial_state_debug": initial_single_obs.tolist()}

    def render(self, mode: str = 'human'):
        scale = 1e-3 
        if self.viewer is None:
            try:
                self.viewer = Viewer(1280,720,headless=(mode=='rgb_array'))
                mesh_vao = load_mesh(self.viewer.ctx, self.viewer.prog, "737.obj") 
                self.aircraft_render_obj = RenderObject(mesh_vao)
                self.aircraft_render_obj.transform.scale = 1.0 / 100.0 # Adjusted scale
                self.aircraft_render_obj.color = (0.1, 0.1, 0.5)
                self.viewer.objects.append(self.aircraft_render_obj)
                self.viewer.objects.append(Grid(self.viewer.ctx,self.viewer.unlit_prog,41,2.0)) # Wider grid
            except Exception as e: self.viewer=None; print(f"Render init err: {e}"); return None
        if self.viewer is None: return None
        
        last_raw = self._get_raw_jsbsim_state()
        lat,lon,alt_msl = last_raw[IDX_LAT_RAD],last_raw[IDX_LON_RAD],last_raw[IDX_ALT_MSL_FT]
        phi,theta,psi = normalize_angle_mpi_pi(last_raw[IDX_ROLL_RAD]), \
                        normalize_angle_mpi_pi(last_raw[IDX_PITCH_RAD]), \
                        normalize_angle_0_2pi(last_raw[IDX_HEADING_RAD])
        
        e,n,u = geodetic_to_enu(lat,lon,alt_msl*FEET_TO_METERS, self.runway_lat_rad,self.runway_lon_rad,self.runway_alt_m)
        self.aircraft_render_obj.transform.position = np.array([e*scale, u*scale, n*scale], dtype=np.float32)
        q_world = Quaternion.from_euler(phi,theta,psi,mode='XYZ') 
        self.aircraft_render_obj.transform.rotation=Quaternion(q_world.w,-q_world.y,-q_world.z,q_world.x)
        
        ac_pos = self.aircraft_render_obj.transform.position
        fwd_vec = self.aircraft_render_obj.transform.rotation * np.array([0,0,1]) # This uses the q_display transformed rotation
        up_vec = self.aircraft_render_obj.transform.rotation * np.array([0,1,0])  # Same here
        
        # Camera behind and slightly above aircraft, looking along its Z body axis (forward)
        cam_offset_behind = 400 * scale # Further back for airliner
        cam_offset_above = 80 * scale   # Higher up
        cam_eye = ac_pos - fwd_vec * cam_offset_behind + up_vec * cam_offset_above
        cam_target = ac_pos + fwd_vec * (500*scale) # Look further ahead
        
        self.viewer.set_view_look_at(cam_eye, cam_target, np.array([0,1,0], dtype=np.float32)) # Use global up for camera

        self.viewer.render()
        if mode=='rgb_array': return self.viewer.get_frame()
        elif mode=='human' and hasattr(self.viewer, 'show') and callable(self.viewer.show) : self.viewer.show(); return None # Added check for show
        return None

    def close(self):
        if self.viewer: self.viewer.close(); self.viewer = None

gymnasium.register(
    id="Boeing737ILSEnv-v0",
    entry_point="jsbsim_gym.ils_env:Boeing737ILSEnv",
    max_episode_steps=2000,
)

if __name__ == "__main__":
    from time import sleep
    print("Creating and testing Boeing737ILSEnv environment...")
    try:
        env = Boeing737ILSEnv(jsbsim_root='.', render_mode=None) # Test without render first
        # env = gymnasium.make("Boeing737ILSEnv-v0", jsbsim_root='.', render_mode='human') # For gym.make
        
        print(f"Obs Space: {env.observation_space.shape}, Act Space: {env.action_space.shape}")
        for ep in range(3): # Test a few episodes
            print(f"\n--- Episode {ep + 1} ---")
            obs, info = env.reset(seed=42 + ep)
            print(f"Initial single obs (for debug): {info.get('initial_state_debug')}")
            term, trunc, tot_rew, ep_steps = False, False, 0, 0
            test_action = np.array([0.0, 0.01, 0.0, 0.4], dtype=np.float32) # Gentle climb, maintain heading

            for i in range(env.max_episode_steps + 50):
                action = test_action 
                obs, reward, term, trunc, info_step = env.step(action)
                tot_rew += reward; ep_steps += 1
                if ep_steps % 200 == 0 or term or trunc: # Print less often
                    print(f"S {ep_steps}, R: {reward:.3f}, TR: {tot_rew:.3f}, Term: {term}, Trunc: {trunc}, FailReason: {info_step.get('failure_reason')}")
                    # print(f"  Current Obs (single frame sample): {obs[-1][7]:.1f} AGL, {obs[-1][11]:.1f} NM, Loc:{obs[-1][0]:.1f}, GS:{obs[-1][1]:.1f}, IAS_err:{obs[-1][2]:.1f}")
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