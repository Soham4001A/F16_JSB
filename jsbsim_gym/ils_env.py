import jsbsim
#import gym #TESTING OLD GYM FOR RENDERING
import gymnasium
import gymnasium as gym
import numpy as np
import collections
import math

# Assuming visualization modules are in a 'visualization' subdirectory relative to this file.
# We might need to adapt the renderer later if the goal visualization needs to change (e.g., show runway).
from .visualization.rendering import Viewer, load_mesh, load_shader, RenderObject, Grid # Keep for now
from .visualization.quaternion import Quaternion # Keep for now

# --- NEW: Constants for ILS Environment ---
# JSBSim properties for the F-16 ILS approach
# We'll select these carefully for our observation space and internal calculations.
ILS_STATE_PROPERTIES = [
    # Position & Orientation (will be processed)
    "position/lat-geod-rad",        # Latitude (geodetic) in radians
    "position/lon-geoc-rad",        # Longitude (geocentric, JSBSim often uses this) in radians
    "position/h-sl-ft",             # Altitude above Sea Level in feet
    "position/h-agl-ft",            # Altitude Above Ground Level in feet (CRITICAL for landing)
    "attitude/phi-rad",             # Roll angle in radians
    "attitude/theta-rad",           # Pitch angle in radians
    "attitude/psi-rad",             # True Heading angle in radians

    # Speeds & Rates
    "velocities/vias-kts",          # Indicated Airspeed in knots (commonly used by pilots)
    "velocities/h-dot-fps",         # Vertical speed in feet per second
    "aero/alpha-deg",               # Angle of Attack in degrees (IMPORTANT for F-16)
    "velocities/p-rad_sec",         # Roll rate in radians per second
    "velocities/q-rad_sec",         # Pitch rate in radians per second
    "velocities/r-rad_sec",         # Yaw rate in radians per second

    # Gear Status (will need to check exact property names for your F-16 model)
    "gear/wow/WOW_nose",            # Weight on Wheels - Nose (e.g., 0 or 1, or force)
    "gear/wow/WOW_left_main",       # Weight on Wheels - Left Main
    "gear/wow/WOW_right_main",      # Weight on Wheels - Right Main

    # Potentially: Engine params if needed for fine throttle control / spool up time, but vias-kts might be enough
    # "propulsion/engine/rpm"
]

# Indices for key properties within the raw_jsbsim_state vector
IDX_LAT_RAD = ILS_STATE_PROPERTIES.index("position/lat-geod-rad")
IDX_LON_RAD = ILS_STATE_PROPERTIES.index("position/lon-geoc-rad")
IDX_ALT_MSL_FT = ILS_STATE_PROPERTIES.index("position/h-sl-ft")
IDX_ALT_AGL_FT = ILS_STATE_PROPERTIES.index("position/h-agl-ft")
IDX_ROLL_RAD = ILS_STATE_PROPERTIES.index("attitude/phi-rad")
IDX_PITCH_RAD = ILS_STATE_PROPERTIES.index("attitude/theta-rad")
IDX_HEADING_RAD = ILS_STATE_PROPERTIES.index("attitude/psi-rad")
IDX_VIAS_KTS = ILS_STATE_PROPERTIES.index("velocities/vias-kts")
IDX_VS_FPS = ILS_STATE_PROPERTIES.index("velocities/h-dot-fps")
IDX_AOA_DEG = ILS_STATE_PROPERTIES.index("aero/alpha-deg")
IDX_ROLL_RATE_RAD_S = ILS_STATE_PROPERTIES.index("velocities/p-rad_sec")
IDX_PITCH_RATE_RAD_S = ILS_STATE_PROPERTIES.index("velocities/q-rad_sec")
IDX_YAW_RATE_RAD_S = ILS_STATE_PROPERTIES.index("velocities/r-rad_sec")
IDX_WOW_NOSE = ILS_STATE_PROPERTIES.index("gear/wow/WOW_nose")
IDX_WOW_LEFT_MAIN = ILS_STATE_PROPERTIES.index("gear/wow/WOW_left_main")
IDX_WOW_RIGHT_MAIN = ILS_STATE_PROPERTIES.index("gear/wow/WOW_right_main")


# Observation space feature names (for clarity, matches our discussion)
OBS_FEATURE_NAMES = [
    "delta_localizer_deg",      # Angular deviation from runway centerline
    "delta_glideslope_deg",     # Angular deviation from glideslope path
    "airspeed_error_kts",       # Current IAS - Target Approach IAS
    "vertical_speed_fps",
    "pitch_angle_rad",
    "roll_angle_rad",
    "heading_error_rad",        # Aircraft True Heading - Runway True Heading
    "altitude_agl_ft",          # Height above ground
    "alpha_deg",                # Angle of Attack
    "pitch_rate_rad_s",
    "roll_rate_rad_s",
    "distance_to_threshold_nm", # Distance to touchdown point
    # Optional: "is_main_gear_on_ground", "is_nose_gear_on_ground" (or derive in agent)
]
NUM_OBS_FEATURES = len(OBS_FEATURE_NAMES)

# Small epsilon for observation space bounds
EPSILON = 1e-5

# Define approximate bounds for the observation features (for normalization or clipping if needed)
# These will need tuning based on expected operational ranges.
OBS_LOW = np.array([
    -10.0,       # delta_localizer_deg (e.g., +/-5 full scale deflection, allow more)
    -5.0,        # delta_glideslope_deg (e.g., +/-2 full scale, allow more)
    -50.0,       # airspeed_error_kts
    -100.0,      # vertical_speed_fps (e.g. -6000 fpm)
    -np.pi/2,    # pitch_angle_rad
    -np.pi,      # roll_angle_rad
    -np.pi,      # heading_error_rad
    -10.0,       # altitude_agl_ft (can be slightly negative if crashed)
    -5.0,        # alpha_deg
    -np.pi,      # pitch_rate_rad_s
    -np.pi,      # roll_rate_rad_s
    0.0,         # distance_to_threshold_nm (min is at threshold)
], dtype=np.float32)

OBS_HIGH = np.array([
    10.0,        # delta_localizer_deg
    5.0,         # delta_glideslope_deg
    50.0,        # airspeed_error_kts
    30.0,        # vertical_speed_fps (e.g. +1800 fpm)
    np.pi/2,     # pitch_angle_rad
    np.pi,       # roll_angle_rad
    np.pi,       # heading_error_rad
    10000.0,     # altitude_agl_ft (max start altitude)
    25.0,        # alpha_deg (F-16 stall AoA might be around 20-25 clean)
    np.pi,       # pitch_rate_rad_s
    np.pi,       # roll_rate_rad_s
    20.0,        # distance_to_threshold_nm (max start distance)
], dtype=np.float32)

# Mean Earth radius in meters
EARTH_RADIUS_METERS = 6.3781e6
# Conversions
FEET_TO_METERS = 0.3048
METERS_TO_FEET = 1.0 / FEET_TO_METERS
KNOTS_TO_MPS = 0.514444
MPS_TO_KNOTS = 1.0 / KNOTS_TO_MPS
NM_TO_METERS = 1852.0
METERS_TO_NM = 1.0 / NM_TO_METERS
RADIANS_TO_DEGREES = 180.0 / np.pi
DEGREES_TO_RADIANS = np.pi / 180.0

# Number of consecutive observation frames stacked
NUM_STACKED_FRAMES = 4 # Reduced from 10 for faster learning, can be tuned

# --- Helper Functions (some from your original, some new) ---
def normalize_angle_mpi_pi(angle_rad: float) -> float:
    if np.isnan(angle_rad) or np.isinf(angle_rad): return 0.0
    angle_rad = angle_rad % (2 * np.pi)
    if angle_rad >= np.pi: angle_rad -= 2 * np.pi
    return angle_rad

def normalize_angle_0_2pi(angle_rad: float) -> float:
    if np.isnan(angle_rad) or np.isinf(angle_rad): return 0.0
    return angle_rad % (2 * np.pi)

def geodetic_to_enu(lat_rad, lon_rad, alt_m, ref_lat_rad, ref_lon_rad, ref_alt_m):
    """Converts Geodetic (lat, lon, alt) to local ENU (East, North, Up) coordinates."""
    # More accurate conversions exist (e.g., using ECEF as intermediate)
    # This is a simplified version assuming small distances.
    dx = EARTH_RADIUS_METERS * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
    dy = EARTH_RADIUS_METERS * (lat_rad - ref_lat_rad)
    dz = alt_m - ref_alt_m
    return dx, dy, dz # East, North, Up

# --- Main Environment Class: F16ILSEnv ---
class F16ILSEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self,
                jsbsim_root: str = '.',
                aircraft_model: str = 'f16',
                dt_hz: int = 10,
                runway_lat_deg: float = 34.0,
                runway_lon_deg: float = -118.0,
                runway_alt_ft: float = 2300.0,
                runway_hdg_deg: float = 220.0,
                glideslope_deg: float = 3.0,
                target_approach_ias_kts: float = 140.0,
                flare_start_agl_ft: float = 50.0,
                max_episode_steps: int = 1500
                ):
        super().__init__()

        self.jsbsim_root = jsbsim_root
        self.aircraft_model = aircraft_model
        self.dt_secs = 1.0 / dt_hz
        self.down_sample = max(1, int(round(60 / dt_hz))) # JSBSim typically runs at 60Hz internally for this model

        self.runway_lat_rad = runway_lat_deg * DEGREES_TO_RADIANS
        self.runway_lon_rad = runway_lon_deg * DEGREES_TO_RADIANS
        self.runway_alt_m = runway_alt_ft * FEET_TO_METERS
        self.runway_hdg_rad = normalize_angle_0_2pi(runway_hdg_deg * DEGREES_TO_RADIANS)
        self.glideslope_rad = glideslope_deg * DEGREES_TO_RADIANS
        self.target_approach_ias_kts = target_approach_ias_kts
        self.flare_start_agl_ft = flare_start_agl_ft
        self.num_stacked_frames = NUM_STACKED_FRAMES


        self.obs_buffer = collections.deque(maxlen=self.num_stacked_frames)
        self.observation_space = gym.spaces.Box(
            low=np.tile(OBS_LOW, (self.num_stacked_frames, 1)),
            high=np.tile(OBS_HIGH, (self.num_stacked_frames, 1)),
            shape=(self.num_stacked_frames, NUM_OBS_FEATURES),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, 0.05], dtype=np.float32),
            high=np.array([1, 1, 1, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        self.simulation = jsbsim.FGFDMExec(self.jsbsim_root, None)
        self.simulation.set_debug_level(0)
        print(f"[INIT] Loading model: {self.aircraft_model}")
        if not self.simulation.load_model(self.aircraft_model):
            raise RuntimeError(f"Failed to load JSBSim model: {self.aircraft_model}")
        
        self.simulation.set_dt(self.dt_secs / self.down_sample)

        # --- Call a simplified initial setup here, similar to NavEnv ---
        print("[INIT] Performing initial JSBSim setup (engine, basic speed/alt)...")
        self.simulation.set_property_value('propulsion/set-running', -1)
        self.simulation.set_property_value("propulsion/active_engine", True)
        self.simulation.set_property_value("propulsion/engine/set_running", 1)
        self.simulation.set_property_value("fcs/throttle-cmd-norm", 0.7) # Ensure some throttle for startup

        # Set a high initial speed and reasonable altitude (like NavEnv, but without specific lat/lon)
        # This helps JSBSim to perhaps initialize its aerodynamics and engine state more robustly
        # We will override these with specific ILS conditions in _set_initial_conditions_for_ils()
        self.simulation.set_property_value('ic/u-fps', 700.0) # ~400 kts TAS
        self.simulation.set_property_value('ic/h-sl-ft', 10000.0) # A generic high altitude
        # DO NOT set lat/lon here, let JSBSim use its default for this initial run_ic
        self.simulation.set_property_value('ic/psi-true-deg', 0.0) # Default heading
        self.simulation.set_property_value('ic/phi-deg', 0.0)
        self.simulation.set_property_value('ic/theta-deg', 0.0)


        print("[INIT] Running initial run_ic()...")
        if not self.simulation.run_ic():
            raise RuntimeError("JSBSim failed on initial run_ic() in __init__.")

        # Run a few steps to let the engine potentially spool up from this initial IC
        print("[INIT] Initial stabilization loop...")
        for i in range(30): # Let it run a bit longer
            self.simulation.run()
            if i % 10 == 0:
                ias = self.simulation.get_property_value("velocities/vias-kts")
                rpm_n1 = self.simulation.get_property_value("propulsion/engine/n1") # Check N1/N2
                rpm_n2 = self.simulation.get_property_value("propulsion/engine/n2")
                lon_check = self.simulation.get_property_value("position/lon-geoc-rad") * RADIANS_TO_DEGREES
                print(f"[INIT] Stab step {i}: IAS={ias:.1f}, N1={rpm_n1:.1f}, N2={rpm_n2:.1f}, Lon={lon_check:.4f}")

        # Now, check the state JSBSim has settled into
        initial_settled_lon = self.simulation.get_property_value("position/lon-geoc-rad") * RADIANS_TO_DEGREES
        initial_settled_ias = self.simulation.get_property_value("velocities/vias-kts")
        print(f"[INIT] State after initial setup: IAS={initial_settled_ias:.1f}, Longitude={initial_settled_lon:.4f}")
        # If initial_settled_lon is NOT 0.0 here, it means JSBSim defaults to a non-zero longitude.

        self.current_step_in_episode = 0
        self.max_episode_steps = max_episode_steps
        self._reset_landing_state_vars()
        self.viewer = None
        self.last_delta_loc = 0
        self.last_delta_gs = 0
        self.last_airspeed_error = 0


    def _reset_landing_state_vars(self):
        """Resets variables that track landing sequence progress."""
        self.main_gear_contact = False
        self.nose_gear_contact = False
        self.all_gear_contact = False # Both mains and nose
        self.main_gear_touchdown_time = -1.0
        self.nose_gear_touchdown_time = -1.0
        self.pitch_at_main_touchdown_rad = 0.0
        self.vs_at_main_touchdown_fps = 0.0
        self.roll_at_main_touchdown_rad = 0.0
        self.loc_dev_at_main_touchdown_deg = 0.0
        self.is_flare_active = False
        self.successfully_landed = False


    def _set_initial_conditions_for_ils(self, rng): # Renamed from _set_initial_conditions
        print("[ILS DEBUG - RESET00 STYLE] Setting ICs like reset00.xml...")

        # Engine Start (critical to get N1/N2 up)
        self.simulation.set_property_value('propulsion/set-running', -1)
        self.simulation.set_property_value("propulsion/active_engine", True)
        self.simulation.set_property_value("propulsion/engine/set_running", 1)
        self.simulation.set_property_value("fcs/throttle-cmd-norm", 0.4) # Ensure decent throttle
        self.simulation.set_property_value("propulsion/engine/n2", 65.0) # Try explicitly setting N2 to a bit above idle
                                                                    # Based on F100-PW-229's <idlen2>60.0</idlen2>

        # Brakes off
        self.simulation.set_property_value("fcs/brake-parking-cmd-norm", 0.0)
        self.simulation.set_property_value("fcs/center-brake-cmd-norm", 0.0)
        self.simulation.set_property_value("fcs/left-brake-cmd-norm", 0.0)
        self.simulation.set_property_value("fcs/right-brake-cmd-norm", 0.0)


        # Speeds (set to a small u-fps to overcome static friction)
        self.simulation.set_property_value('ic/u-fps', 10.0) # Small forward nudge
        self.simulation.set_property_value('ic/v-fps', 0.0)
        self.simulation.set_property_value('ic/w-fps', 0.0)
        # vias-kts will be calculated, but setting a target can sometimes help guide the IC
        self.simulation.set_property_value('ic/vias-kts', 5.0)


        # Position (still attempting to set desired ILS start)
        target_lat_deg = self.runway_lat_rad * RADIANS_TO_DEGREES # Simplified for this test
        target_lon_deg = self.runway_lon_rad * RADIANS_TO_DEGREES # e.g., -118.0
        target_alt_ft = self.runway_alt_m * METERS_TO_FEET + 10.0 # On ground for this test

        print(f"[ILS DEBUG - ENGINE/BRAKE TEST] Attempting: lat={target_lat_deg:.2f}, lon={target_lon_deg:.2f}, alt={target_alt_ft:.1f}")
        self.simulation.set_property_value('ic/lat-geod-deg', target_lat_deg)
        self.simulation.set_property_value('ic/lon-geod-deg', target_lon_deg)
        self.simulation.set_property_value('ic/h-sl-ft', target_alt_ft)

        # Orientation
        self.simulation.set_property_value('ic/phi-deg', 0.0)
        self.simulation.set_property_value('ic/theta-deg', 0.0) 
        self.simulation.set_property_value('ic/psi-true-deg', self.runway_hdg_rad * RADIANS_TO_DEGREES)

        self.simulation.set_property_value("fcs/flap-cmd-norm", 0.0) # Flaps up for ground test
        self.simulation.set_property_value("gear/gear-cmd-norm", 1.0) # Gear down
        self.simulation.set_property_value("fcs/speedbrake-cmd-norm", 0.0)


        print("[ILS DEBUG - ENGINE/BRAKE TEST] Running run_ic()...")
        if not self.simulation.run_ic():
            print("[ILS DEBUG ERROR - ENGINE/BRAKE TEST] run_ic() failed.")

        print("[ILS DEBUG - ENGINE/BRAKE TEST] Stabilization loop...")
        for i in range(60): 
            self.simulation.run()
            if i % 10 == 0:
                ias = self.simulation.get_property_value("velocities/vias-kts")
                n1 = self.simulation.get_property_value("propulsion/engine/n1")
                n2 = self.simulation.get_property_value("propulsion/engine/n2")
                thrust = self.simulation.get_property_value("propulsion/engine/thrust-lbs")
                lon_check = self.simulation.get_property_value("position/lon-geoc-rad") * RADIANS_TO_DEGREES
                print(f"[ILS DEBUG] Stab step {i}: IAS={ias:.1f}, N1={n1:.1f}, N2={n2:.1f}, Thrust={thrust:.1f}, Lon={lon_check:.4f}")


        # --- VERIFY ---
        # (Same verification prints as before)
        current_raw_state_after_ic = self._get_raw_jsbsim_state()
        actual_lat_deg = current_raw_state_after_ic[IDX_LAT_RAD] * RADIANS_TO_DEGREES
        actual_lon_deg = current_raw_state_after_ic[IDX_LON_RAD] * RADIANS_TO_DEGREES # This is crucial
        # ... rest of verification ...
        print(f"[ILS DEBUG - RESET00 STYLE] Actual state: lat={actual_lat_deg:.4f}, lon={actual_lon_deg:.4f} ...")
        # ... print distance ...

        # IF THIS STILL FAILS (lon=0, ias=0):
        # The issue is deeply rooted in this F-16 model's interaction with Python API's IC setting.
        # The fact that NavEnv *doesn't* set lat/lon and it works is the biggest clue.
        # It means NavEnv is using JSBSim's default positioning for that F-16,
        # which likely comes from an internal default in f16.xml or an associated file
        # that correctly sets a non-zero longitude.

        # If after this, lon is still 0.0:
        # The next step would be to accept JSBSim's default starting position (by *not* setting ic/lat or ic/lon)
        # and then make your RL agent fly from that default starting point to your ILS intercept.
        # This means:
        # 1. In _set_initial_conditions_for_ils:
        #    - DO NOT set 'ic/lat-geod-deg' or 'ic/lon-geod-deg'.
        #    - DO set 'ic/h-sl-ft' to your desired starting altitude for the ILS INTERCEPT (e.g., 3000ft AGL near the start of the localizer).
        #    - DO set 'ic/u-fps' to your desired approach speed.
        #    - DO set 'ic/psi-true-deg' to roughly align with the runway.
        #    - Engine running, flaps, gear.
        # 2. Then `run_ic()`.
        # 3. JSBSim will place you at its default Lat/Lon but hopefully at your specified Alt/Speed/Heading.
        # 4. Your observation space's distance_to_threshold_nm etc. will be large initially.
        # 5. Your agent learns to navigate from there. This is harder but might be unavoidable if direct positioning fails.

    def _get_raw_jsbsim_state(self) -> np.ndarray:
        """Reads the defined properties from JSBSim."""
        state_values = np.zeros(len(ILS_STATE_PROPERTIES), dtype=np.float32)
        for i, prop_name in enumerate(ILS_STATE_PROPERTIES):
            try:
                val = self.simulation.get_property_value(prop_name)
                # Handle potential None or non-numeric values from JSBSim if property missing or error
                if val is None or np.isnan(val) or np.isinf(val):
                    # print(f"Warning: JSBSim property '{prop_name}' returned problematic value: {val}. Using 0.")
                    val = 0.0
                state_values[i] = float(val)
            except Exception as e:
                # print(f"Warning: Could not retrieve JSBSim property '{prop_name}': {e}. Using 0.")
                state_values[i] = 0.0 # Default to 0 if property is problematic
        return state_values

    def _calculate_ils_deviations_and_distances(self, raw_state: np.ndarray):
        """Calculates ILS deviations based on current aircraft state and runway."""
        ac_lat_rad = raw_state[IDX_LAT_RAD]
        ac_lon_rad = raw_state[IDX_LON_RAD]
        ac_alt_msl_ft = raw_state[IDX_ALT_MSL_FT]
        ac_hdg_rad = normalize_angle_0_2pi(raw_state[IDX_HEADING_RAD])

        # Convert A/C position to local ENU relative to runway threshold
        # Note: JSBSim lat/lon might need conversion if one is geodetic and other geocentric
        # Assuming raw_state[IDX_LAT_RAD] (geodetic) is compatible with runway_lat_rad (geodetic)
        east_m, north_m, up_m = geodetic_to_enu(
            ac_lat_rad, ac_lon_rad, ac_alt_msl_ft * FEET_TO_METERS,
            self.runway_lat_rad, self.runway_lon_rad, self.runway_alt_m
        )

        # Distance and bearing to threshold (2D projection)
        dist_to_thresh_m_2d = math.sqrt(east_m**2 + north_m**2)
        dist_to_thresh_nm = dist_to_thresh_m_2d * METERS_TO_NM

        # --- Localizer Deviation ---
        # Vector from threshold to aircraft in ENU plane: (east_m, north_m)
        # Runway direction vector in ENU plane: (sin(runway_hdg), cos(runway_hdg))
        runway_dir_east = np.sin(self.runway_hdg_rad)
        runway_dir_north = np.cos(self.runway_hdg_rad)

        # Lateral distance from extended centerline (cross product in 2D, or dot product with perpendicular)
        # Perpendicular vector to runway: (-cos(runway_hdg), sin(runway_hdg)) or (cos(runway_hdg), -sin(runway_hdg))
        # Let's use dot product of AC_pos vector with runway_perpendicular vector.
        # Vector perpendicular to runway (pointing right of runway): (cos(hdg), -sin(hdg))
        runway_perp_east = np.cos(self.runway_hdg_rad)
        runway_perp_north = -np.sin(self.runway_hdg_rad)

        lateral_dist_m = east_m * runway_perp_east + north_m * runway_perp_north
        # longitudinal_dist_m = east_m * runway_dir_east + north_m * runway_dir_north # Distance along centerline from threshold

        # Angular deviation (small angle approx: lateral_dist / longitudinal_dist_from_transmitter)
        # Transmitter is usually near runway stop end. For simplicity, use dist_to_thresh_m_2d
        # Ensure dist_to_thresh_m_2d is not zero to avoid division by zero
        if dist_to_thresh_m_2d > 1.0: # Min 1 meter to avoid div by zero
             delta_loc_rad = math.atan2(lateral_dist_m, dist_to_thresh_m_2d) # More robust
            # delta_loc_rad = lateral_dist_m / dist_to_thresh_m_2d # Simpler approx.
        else:
            delta_loc_rad = 0.0
        delta_loc_deg = delta_loc_rad * RADIANS_TO_DEGREES

        # --- Glideslope Deviation ---
        ac_alt_agl_ft = raw_state[IDX_ALT_AGL_FT]
        # Desired altitude AGL on glideslope at current longitudinal distance
        # Use dist_to_thresh_m_2d as approximation for longitudinal distance
        desired_alt_agl_on_gs_m = dist_to_thresh_m_2d * np.tan(self.glideslope_rad)
        desired_alt_agl_on_gs_ft = desired_alt_agl_on_gs_m * METERS_TO_FEET

        alt_diff_ft = ac_alt_agl_ft - desired_alt_agl_on_gs_ft

        if dist_to_thresh_m_2d > 1.0:
            delta_gs_rad = math.atan2(alt_diff_ft * FEET_TO_METERS, dist_to_thresh_m_2d) # More robust
            # delta_gs_rad = (alt_diff_ft * FEET_TO_METERS) / dist_to_thresh_m_2d # Simpler approx.
        else:
            delta_gs_rad = 0.0
        delta_gs_deg = delta_gs_rad * RADIANS_TO_DEGREES

        # --- Heading Error ---
        heading_error_rad = normalize_angle_mpi_pi(ac_hdg_rad - self.runway_hdg_rad)

        return delta_loc_deg, delta_gs_deg, heading_error_rad, dist_to_thresh_nm


    def _get_observation(self, raw_state: np.ndarray):
        """Constructs the observation vector from raw JSBSim state and ILS calculations."""
        delta_loc_deg, delta_gs_deg, heading_error_rad, dist_to_thresh_nm = \
            self._calculate_ils_deviations_and_distances(raw_state)

        # Store for reward shaping
        self.last_delta_loc = delta_loc_deg
        self.last_delta_gs = delta_gs_deg
        current_ias_kts = raw_state[IDX_VIAS_KTS]
        self.last_airspeed_error = current_ias_kts - self.target_approach_ias_kts


        obs_features = np.array([
            delta_loc_deg,
            delta_gs_deg,
            current_ias_kts - self.target_approach_ias_kts, # airspeed_error_kts
            raw_state[IDX_VS_FPS],                          # vertical_speed_fps
            normalize_angle_mpi_pi(raw_state[IDX_PITCH_RAD]), # pitch_angle_rad
            normalize_angle_mpi_pi(raw_state[IDX_ROLL_RAD]),  # roll_angle_rad
            heading_error_rad,                              # heading_error_rad
            raw_state[IDX_ALT_AGL_FT],                      # altitude_agl_ft
            raw_state[IDX_AOA_DEG],                         # alpha_deg
            raw_state[IDX_PITCH_RATE_RAD_S],                # pitch_rate_rad_s
            raw_state[IDX_ROLL_RATE_RAD_S],                 # roll_rate_rad_s
            dist_to_thresh_nm                               # distance_to_threshold_nm
        ], dtype=np.float32)
        
        assert len(obs_features) == NUM_OBS_FEATURES, \
            f"Mismatch: _get_observation produced {len(obs_features)} features, expected {NUM_OBS_FEATURES}"

        # Clip observation to defined bounds (helps with stability if some values go wild)
        # return np.clip(obs_features, OBS_LOW, OBS_HIGH)
        return obs_features # Or let the agent/normalization layers handle this

    def _update_landing_gear_status(self, raw_state: np.ndarray):
        """Updates internal variables tracking gear contact."""
        # Threshold for WOW sensors (JSBSim might output small non-zero values even if not firmly on ground)
        WOW_THRESHOLD = 0.5 # Adjust based on F-16 model's WOW output

        left_main_on_ground = raw_state[IDX_WOW_LEFT_MAIN] > WOW_THRESHOLD
        right_main_on_ground = raw_state[IDX_WOW_RIGHT_MAIN] > WOW_THRESHOLD
        nose_on_ground = raw_state[IDX_WOW_NOSE] > WOW_THRESHOLD

        current_time = self.current_step_in_episode * self.dt_secs

        if not self.main_gear_contact and left_main_on_ground and right_main_on_ground:
            self.main_gear_contact = True
            self.main_gear_touchdown_time = current_time
            self.pitch_at_main_touchdown_rad = normalize_angle_mpi_pi(raw_state[IDX_PITCH_RAD])
            self.vs_at_main_touchdown_fps = raw_state[IDX_VS_FPS]
            self.roll_at_main_touchdown_rad = normalize_angle_mpi_pi(raw_state[IDX_ROLL_RAD])
            delta_loc, _, _, _ = self._calculate_ils_deviations_and_distances(raw_state)
            self.loc_dev_at_main_touchdown_deg = delta_loc


        if not self.nose_gear_contact and nose_on_ground:
            self.nose_gear_contact = True
            self.nose_gear_touchdown_time = current_time

        if self.main_gear_contact and self.nose_gear_contact:
            self.all_gear_contact = True


    def _calculate_reward(self, raw_state: np.ndarray, current_obs_features: np.ndarray):
        """Calculates the reward for the current state."""
        reward = 0.0

        # Unpack features needed for reward
        delta_loc_deg = current_obs_features[OBS_FEATURE_NAMES.index("delta_localizer_deg")]
        delta_gs_deg = current_obs_features[OBS_FEATURE_NAMES.index("delta_glideslope_deg")]
        airspeed_error_kts = current_obs_features[OBS_FEATURE_NAMES.index("airspeed_error_kts")]
        roll_angle_rad = current_obs_features[OBS_FEATURE_NAMES.index("roll_angle_rad")]
        pitch_angle_rad = current_obs_features[OBS_FEATURE_NAMES.index("pitch_angle_rad")]
        aoa_deg = current_obs_features[OBS_FEATURE_NAMES.index("alpha_deg")]
        alt_agl_ft = current_obs_features[OBS_FEATURE_NAMES.index("altitude_agl_ft")]
        dist_to_thresh_nm = current_obs_features[OBS_FEATURE_NAMES.index("distance_to_threshold_nm")]

        # --- Path Adherence Rewards (Active throughout, especially before flare) ---
        # Localizer
        loc_tolerance_deg = 0.5
        if abs(delta_loc_deg) < loc_tolerance_deg:
            reward += 0.1 # Constant reward for being on loc
        else:
            reward -= 0.1 * (abs(delta_loc_deg) - loc_tolerance_deg)**2 # Penalize deviation

        # Glideslope (less penalty during active flare)
        gs_tolerance_deg = 0.25
        if not self.is_flare_active:
            if abs(delta_gs_deg) < gs_tolerance_deg:
                reward += 0.1 # Constant reward for being on gs
            else:
                reward -= 0.1 * (abs(delta_gs_deg) - gs_tolerance_deg)**2

        # Airspeed
        speed_tolerance_kts = 5.0
        if abs(airspeed_error_kts) < speed_tolerance_kts:
            reward += 0.05
        else:
            reward -= 0.02 * (abs(airspeed_error_kts) - speed_tolerance_kts)**2

        # Stability
        reward -= 0.05 * (abs(roll_angle_rad) / (np.pi/6))**2  # Penalize excessive roll (e.g. > 30 deg)
        if alt_agl_ft > self.flare_start_agl_ft + 20: # Only penalize pitch outside flare
             if abs(pitch_angle_rad) > 15 * DEGREES_TO_RADIANS : # e.g. > 15 deg pitch
                reward -= 0.1 * (abs(pitch_angle_rad) - 15 * DEGREES_TO_RADIANS)**2

        # AoA (penalize if too high and not in flare/touchdown attitude)
        target_approach_aoa = 8.0 # Example, F-16 specific
        max_aoa_approach = 13.0
        if not self.is_flare_active and aoa_deg > max_aoa_approach:
            reward -= 0.1 * (aoa_deg - max_aoa_approach)**2


        # --- Flare Logic ---
        if alt_agl_ft <= self.flare_start_agl_ft and not self.main_gear_contact:
            self.is_flare_active = True

        if self.is_flare_active and not self.main_gear_contact:
            # Reward for reducing sink rate
            vs_fps = raw_state[IDX_VS_FPS]
            if -7.0 < vs_fps < -0.5: # Gentle sink rate
                reward += 0.1
            elif vs_fps >= -0.5: # Leveling off too high or ballooning
                reward -= 0.1
            else: # Sinking too fast
                reward -= 0.05 * abs(vs_fps - (-7.0))


            # Reward for achieving flare pitch (e.g., 5-8 degrees for F-16, depends on model)
            target_flare_pitch_deg = 6.0
            if abs(pitch_angle_rad * RADIANS_TO_DEGREES - target_flare_pitch_deg) < 2.0:
                reward += 0.1
            else:
                reward -= 0.05 * (abs(pitch_angle_rad * RADIANS_TO_DEGREES - target_flare_pitch_deg) - 2.0)**2


        # --- Touchdown Event Rewards/Penalties (Awarded ONCE implicitly via termination check) ---
        # These are handled more by the termination conditions leading to a large success/failure bonus.
        # But we can add small incentives/penalties here too.

        # Penalty for control effort (optional)
        # reward -= 0.001 * np.sum(np.square(self.last_action))


        # Progress towards runway (before flare)
        if alt_agl_ft > self.flare_start_agl_ft * 1.5 : # Only if well above flare height
            if dist_to_thresh_nm < self.prev_dist_to_thresh_nm:
                 reward += 0.01 * (self.prev_dist_to_thresh_nm - dist_to_thresh_nm) # Small reward for getting closer
            self.prev_dist_to_thresh_nm = dist_to_thresh_nm


        # Keep alive reward (small positive reward for not failing)
        reward += 0.01

        return reward

    def _check_termination_truncation(self, raw_state: np.ndarray, current_obs_features: np.ndarray):
        """Checks for episode termination or truncation conditions."""
        terminated = False
        success = False # Will be set True if successful landing conditions met

        alt_agl_ft = current_obs_features[OBS_FEATURE_NAMES.index("altitude_agl_ft")]
        dist_to_thresh_nm = current_obs_features[OBS_FEATURE_NAMES.index("distance_to_threshold_nm")]
        delta_loc_deg = current_obs_features[OBS_FEATURE_NAMES.index("delta_localizer_deg")]
        delta_gs_deg = current_obs_features[OBS_FEATURE_NAMES.index("delta_glideslope_deg")]
        airspeed_error_kts = current_obs_features[OBS_FEATURE_NAMES.index("airspeed_error_kts")]
        roll_angle_rad = current_obs_features[OBS_FEATURE_NAMES.index("roll_angle_rad")]
        pitch_angle_rad = current_obs_features[OBS_FEATURE_NAMES.index("pitch_angle_rad")]
        aoa_deg = current_obs_features[OBS_FEATURE_NAMES.index("alpha_deg")]

        # Crash conditions
        if alt_agl_ft < -2.0: # Crashed
            # print("TERMINATED: Crashed (AGL < -2 ft)")
            terminated = True; failure_reason = "crashed_low_alt"
        if raw_state[IDX_VS_FPS] < -30 and alt_agl_ft < 10 : # Excessive sink rate close to ground
            # print("TERMINATED: Crashed (Excessive Sink Rate at low alt)")
            terminated = True; failure_reason = "crashed_sink_rate"
        if abs(roll_angle_rad) > 60 * DEGREES_TO_RADIANS and alt_agl_ft < 200 : # Extreme bank angle at low altitude
            # print("TERMINATED: Crashed (Extreme Roll at low alt)")
            terminated = True; failure_reason = "crashed_extreme_roll"
        if abs(pitch_angle_rad) > 45 * DEGREES_TO_RADIANS and alt_agl_ft < 200 : # Extreme pitch
             # print("TERMINATED: Crashed (Extreme Pitch at low alt)")
            terminated = True; failure_reason = "crashed_extreme_pitch"


        # Stall (F-16 specific AoA limits would be better)
        # Max AoA for F-16 can be around 20-25 deg clean, lower with gear/flaps.
        # Let's say stall if AoA > 20 deg for a noticeable duration or very high instantly.
        if aoa_deg > 22.0 and raw_state[IDX_VIAS_KTS] < 100: # Simplified stall
            # print("TERMINATED: Stalled")
            terminated = True; failure_reason = "stalled"

        # Out of bounds / Unstable approach
        # Check these mainly before flare or if way off course
        if not self.is_flare_active and dist_to_thresh_nm > 0.2 : # Don't penalize small deviations during flare/rollout
            if abs(delta_loc_deg) > 7.0 :
                # print("TERMINATED: Too far off localizer")
                terminated = True; failure_reason = "off_loc"
            if abs(delta_gs_deg) > 3.5 :
                 # print("TERMINATED: Too far off glideslope")
                terminated = True; failure_reason = "off_gs"
            if abs(airspeed_error_kts) > 40.0: # Way too fast or slow
                # print("TERMINATED: Airspeed out of approach range")
                terminated = True; failure_reason = "bad_airspeed"

        # Flew past runway without landing
        if dist_to_thresh_nm < -0.5 and not self.all_gear_contact : # 0.5 NM past threshold
            # print("TERMINATED: Flew past runway")
            terminated = True; failure_reason = "flew_past"


        # --- Landing Specific Terminations ---
        if self.main_gear_contact: # Main gear has touched down
            # Hard landing
            MAX_ACCEPTABLE_SINK_FPS = -12.0 # Approx 720 fpm
            if self.vs_at_main_touchdown_fps < MAX_ACCEPTABLE_SINK_FPS:
                # print(f"TERMINATED: Hard Landing (VS: {self.vs_at_main_touchdown_fps:.2f} fps)")
                terminated = True; failure_reason = "hard_landing"

            # Bad pitch at touchdown (tail strike or nose low)
            MIN_PITCH_MAIN_TD_DEG = 1.0 # Min nose up
            MAX_PITCH_MAIN_TD_DEG = 13.0 # Max to avoid tail strike (F-16 specific)
            pitch_at_td_deg = self.pitch_at_main_touchdown_rad * RADIANS_TO_DEGREES
            if not (MIN_PITCH_MAIN_TD_DEG < pitch_at_td_deg < MAX_PITCH_MAIN_TD_DEG):
                # print(f"TERMINATED: Bad pitch at main touchdown ({pitch_at_td_deg:.2f} deg)")
                terminated = True; failure_reason = "bad_pitch_td"

            # Excessive roll at touchdown
            if abs(self.roll_at_main_touchdown_rad) > (7.0 * DEGREES_TO_RADIANS):
                # print(f"TERMINATED: Excessive roll at touchdown ({self.roll_at_main_touchdown_rad * RADIANS_TO_DEGREES:.1f} deg)")
                terminated = True; failure_reason = "roll_at_td"

            # Off centerline at touchdown
            if abs(self.loc_dev_at_main_touchdown_deg) > 1.5 : # e.g. > 1.5 deg off LOC is ~75ft at 1nm, more at threshold
                # print(f"TERMINATED: Off centerline at touchdown ({self.loc_dev_at_main_touchdown_deg:.2f} deg)")
                terminated = True; failure_reason = "off_center_td"


        if self.nose_gear_contact: # Nose gear has touched
            # Nose gear first or simultaneous with mains
            # (allow very small positive diff for "simultaneous" due to discrete time)
            if not self.main_gear_contact or self.nose_gear_touchdown_time < (self.main_gear_touchdown_time - self.dt_secs * 0.5):
                # print("TERMINATED: Nose gear touched before or without main gear")
                terminated = True; failure_reason = "nose_first_td"

        # Successful Landing Condition
        if self.all_gear_contact and not terminated : # All three wheels on ground
            # And main gear touched first, gently, with good pitch, on centerline
            if self.main_gear_touchdown_time < self.nose_gear_touchdown_time and \
               self.vs_at_main_touchdown_fps > MAX_ACCEPTABLE_SINK_FPS and \
               MIN_PITCH_MAIN_TD_DEG < (self.pitch_at_main_touchdown_rad * RADIANS_TO_DEGREES) < MAX_PITCH_MAIN_TD_DEG and \
               abs(self.roll_at_main_touchdown_rad) < (7.0 * DEGREES_TO_RADIANS) and \
               abs(self.loc_dev_at_main_touchdown_deg) < 1.0: # Stricter for success
                # print("TERMINATED: Successful Landing!")
                terminated = True
                success = True
                self.successfully_landed = True
                failure_reason = "none_success" # No failure

        # Truncation
        truncated = False
        if not terminated and self.current_step_in_episode >= self.max_episode_steps:
            # print("TRUNCATED: Max episode steps reached.")
            truncated = True
            failure_reason = "truncated_max_steps"


        info = {'success': success}
        if terminated or truncated:
            info['failure_reason'] = failure_reason if not success else "none_success"
            info['dist_to_threshold_at_end_nm'] = dist_to_thresh_nm
            info['alt_agl_at_end_ft'] = alt_agl_ft
            if self.main_gear_contact:
                info['vs_at_main_td_fps'] = self.vs_at_main_touchdown_fps
                info['pitch_at_main_td_deg'] = self.pitch_at_main_touchdown_rad * RADIANS_TO_DEGREES
                info['roll_at_main_td_deg'] = self.roll_at_main_touchdown_rad * RADIANS_TO_DEGREES
                info['loc_dev_at_main_td_deg'] = self.loc_dev_at_main_touchdown_deg


        return terminated, truncated, info

    def step(self, action: np.ndarray):
        self.current_step_in_episode += 1
        self.last_action = action # Store for potential reward shaping (control effort)

        # Apply action to JSBSim
        self.simulation.set_property_value("fcs/aileron-cmd-norm", float(action[0]))
        self.simulation.set_property_value("fcs/elevator-cmd-norm", float(action[1]))
        self.simulation.set_property_value("fcs/rudder-cmd-norm", float(action[2]))
        self.simulation.set_property_value("fcs/throttle-cmd-norm", float(action[3]))

        for _ in range(self.down_sample):
            self.simulation.run()

        raw_state = self._get_raw_jsbsim_state()
        self._update_landing_gear_status(raw_state)
        current_single_obs_features = self._get_observation(raw_state)
        self.obs_buffer.append(current_single_obs_features)
        stacked_obs = np.array(self.obs_buffer, dtype=np.float32)

        reward = self._calculate_reward(raw_state, current_single_obs_features)
        terminated, truncated, info = self._check_termination_truncation(raw_state, current_single_obs_features)

        if terminated:
            if info.get('success', False):
                reward += 500.0
            else:
                reward -= 200.0
        elif truncated:
            reward -= 100.0

        # --- BEGIN DEBUGGING BLOCK FOR STEP OBSERVATION SPACE ---
        # Ensure self.observation_space is defined in __init__
        if not self.observation_space.contains(stacked_obs):
            print(f"ERROR IN STEP ({self.current_step_in_episode}): Observation is NOT within the defined observation space!")
            for i in range(self.num_stacked_frames): # Or NUM_STACKED_FRAMES if global
                frame = stacked_obs[i]
                # Assuming OBS_LOW, OBS_HIGH are defined globally or as self.OBS_LOW, self.OBS_HIGH
                # And OBS_FEATURE_NAMES is also accessible
                single_frame_space = gymnasium.spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
                if not single_frame_space.contains(frame):
                    print(f"  Problem in stacked frame {i} (latest in buffer is frame {self.num_stacked_frames-1}):")
                    for j, feature_name in enumerate(OBS_FEATURE_NAMES):
                        val = frame[j]
                        low = OBS_LOW[j]
                        high = OBS_HIGH[j]
                        if not (low <= val <= high):
                            print(f"    Feature '{feature_name}' (idx {j}) value: {val:.4f} is OUT OF BOUNDS [{low:.4f}, {high:.4f}]")
            # Uncomment to stop execution immediately when an out-of-bounds observation is detected in step
            # raise ValueError(f"Observation from step() at step {self.current_step_in_episode} is not within observation_space.")
        # --- END DEBUGGING BLOCK FOR STEP OBSERVATION SPACE ---

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

        # Ensure OBS_FEATURE_NAMES is accessible here
        self.prev_dist_to_thresh_nm = initial_single_obs[
            OBS_FEATURE_NAMES.index("distance_to_threshold_nm")
        ] + 1.0

        for _ in range(self.num_stacked_frames): # Or NUM_STACKED_FRAMES if global
            self.obs_buffer.append(np.copy(initial_single_obs))

        initial_stacked_obs = np.array(self.obs_buffer, dtype=np.float32)

        # --- BEGIN DEBUGGING BLOCK FOR RESET OBSERVATION SPACE ---
        # Ensure self.observation_space is defined in __init__
        if not self.observation_space.contains(initial_stacked_obs):
            print("ERROR IN RESET: Initial observation is NOT within the defined observation space!")
            for i in range(self.num_stacked_frames): # Or NUM_STACKED_FRAMES if global
                frame = initial_stacked_obs[i]
                # Assuming OBS_LOW, OBS_HIGH, OBS_FEATURE_NAMES are accessible
                single_frame_space = gymnasium.spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
                if not single_frame_space.contains(frame):
                    print(f"  Problem in stacked frame {i}:")
                    for j, feature_name in enumerate(OBS_FEATURE_NAMES):
                        val = frame[j]
                        low = OBS_LOW[j]
                        high = OBS_HIGH[j]
                        if not (low <= val <= high):
                            print(f"    Feature '{feature_name}' (idx {j}) value: {val:.4f} is OUT OF BOUNDS [{low:.4f}, {high:.4f}]")
            # Uncomment to stop execution immediately when an out-of-bounds observation is detected in reset
            # raise ValueError("Initial observation from reset() is not within observation_space.")
        # --- END DEBUGGING BLOCK FOR RESET OBSERVATION SPACE ---

        return initial_stacked_obs, {"initial_state": initial_single_obs.tolist()}


    def render(self, mode: str = 'human'):
        # --- This render method is largely from your original F16 env ---
        # --- It might need adjustments for visualizing the runway/ILS path ---
        # --- instead of or in addition to the old 'goal cylinder'       ---
        scale = 1e-3

        if self.viewer is None:
            try:
                self.viewer = Viewer(1280, 720, headless=(mode == 'rgb_array')) # Pass headless based on mode

                f16_mesh_vao = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
                self.f16_render_obj = RenderObject(f16_mesh_vao)
                self.f16_render_obj.transform.scale = 1.0 / 30.0
                self.f16_render_obj.color = (0.0, 0.0, 0.4)
                self.viewer.objects.append(self.f16_render_obj)

                # TODO: Add runway visualization instead of/in addition to cylinder.obj
                # For now, let's remove the old goal cylinder if it's not relevant.
                # goal_mesh_vao = load_mesh(self.viewer.ctx, self.viewer.prog, "cylinder.obj")
                # self.goal_render_obj = RenderObject(goal_mesh_vao)
                # self.goal_render_obj.transform.scale = scale * 100.0
                # self.goal_render_obj.color = (0.0, 0.4, 0.0)
                # self.viewer.objects.append(self.goal_render_obj)

                self.viewer.objects.append(Grid(self.viewer.ctx, self.viewer.unlit_prog, 21, 1.0)) # Grid
            except Exception as e:
                print(f"Error initializing renderer or loading assets: {e}")
                self.viewer = None
                return None

        if self.viewer is None: return None

        # Get current aircraft state for rendering (use last raw state for positions)
        last_raw_state = self._get_raw_jsbsim_state() # Get fresh state for rendering
        ac_lat_rad = last_raw_state[IDX_LAT_RAD]
        ac_lon_rad = last_raw_state[IDX_LON_RAD]
        ac_alt_msl_ft = last_raw_state[IDX_ALT_MSL_FT]
        ac_roll_rad = normalize_angle_mpi_pi(last_raw_state[IDX_ROLL_RAD])
        ac_pitch_rad = normalize_angle_mpi_pi(last_raw_state[IDX_PITCH_RAD])
        ac_hdg_rad = normalize_angle_0_2pi(last_raw_state[IDX_HEADING_RAD])


        # Convert aircraft global (lat,lon,alt) to a local rendering frame
        # For simplicity, let's use ENU relative to runway threshold as our rendering origin
        render_east_m, render_north_m, render_up_m = geodetic_to_enu(
            ac_lat_rad, ac_lon_rad, ac_alt_msl_ft * FEET_TO_METERS,
            self.runway_lat_rad, self.runway_lon_rad, self.runway_alt_m
        )

        # Viewer coordinate system: +Y up, +X right, +Z towards camera (or -Z forward)
        # Mapping: Sim East -> Viewer X, Sim North -> Viewer Z (depth), Sim Up -> Viewer Y
        self.f16_render_obj.transform.position = np.array([
            render_east_m * scale,
            render_up_m * scale,
            render_north_m * scale # North into the screen (positive Z)
        ], dtype=np.float32)

        # Euler angles: JSBSim psi (heading) is about Z_down. Viewer yaw is about Y_up.
        # JSBSim phi (roll) is about X_fwd. Viewer roll is about Z_fwd (or X_fwd).
        # JSBSim theta (pitch) is about Y_right. Viewer pitch is about X_right.
        # Need to be careful with Euler angle conventions and render coordinate system.
        # Your original code used: q_display = Quaternion(q_jsb.w, -q_jsb.y, -q_jsb.z, q_jsb.x)
        # This implies a coordinate transformation in the quaternion.
        # Let's try standard ENU to Viewer mapping for angles:
        # Roll (about fwd axis = North for viewer's Z): ac_roll_rad
        # Pitch (about right axis = East for viewer's X): ac_pitch_rad
        # Yaw (about up axis = Up for viewer's Y, maps to heading): ac_hdg_rad
        # Viewer expects: pitch (X), yaw (Y), roll (Z) for its from_euler if mode=1 (ZYX order application)
        # So, Viewer_Yaw from ac_hdg_rad, Viewer_Pitch from ac_pitch_rad, Viewer_Roll from ac_roll_rad
        # The render orientation will be tricky.
        q_aircraft_world = Quaternion.from_euler(ac_roll_rad, ac_pitch_rad, ac_hdg_rad, mode='XYZ') # Assuming XYZ application order
        # Apply the same transformation as your original code for consistency for now
        self.f16_render_obj.transform.rotation = Quaternion(
             q_aircraft_world.w, -q_aircraft_world.y, -q_aircraft_world.z, q_aircraft_world.x
        )


        # Camera: Follow aircraft
        f16_viewer_pos = self.f16_render_obj.transform.position
        # Simple behind-and-above camera:
        # Use aircraft's orientation to offset camera
        fwd_render_vec = self.f16_render_obj.transform.rotation * np.array([0,0,1]) # Aircraft's forward in viewer space
        up_render_vec = self.f16_render_obj.transform.rotation * np.array([0,1,0])   # Aircraft's up in viewer space

        cam_offset_behind = 15.0 * scale # Scaled units
        cam_offset_above = 5.0 * scale

        cam_eye_position = f16_viewer_pos - fwd_render_vec * cam_offset_behind + up_render_vec * cam_offset_above
        cam_target_position = f16_viewer_pos + fwd_render_vec * 10.0 * scale # Look slightly ahead of aircraft

        self.viewer.set_view_look_at(cam_eye_position, cam_target_position, up_render_vec)

        self.viewer.render()

        if mode == 'rgb_array':
            return self.viewer.get_frame()
        elif mode == 'human':
            self.viewer.show() # If your viewer has a show method for non-headless
            return None
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.simulation:
            # JSBSim FGFDMExec doesn't have an explicit close, relies on GC
            pass

# --- End of F16ILSEnv ---

def wrap_jsbsim(**kwargs):
    """
    Factory function to create an instance of JSBSimEnv wrapped with PositionReward.

    Args:
        **kwargs: Arguments to pass to the JSBSimEnv constructor (e.g., `root`).

    Returns:
        PositionReward: The wrapped JSBSim environment.
    """
    env = F16ILSEnv(**kwargs)
    return env

# Register the environment with Gymnasium.
# This allows gym.make("JSBSim-v0") to create the wrapped environment.
gym.register(
    id="F16ILSEnv-v0",
    entry_point="jsbsim_gym.ils_env:wrap_jsbsim", # Assumes this file is jsbsim_gym.py within jsbsim_gym package
    # If this file is top-level, e.g. 'my_env_file.py', entry_point would be 'my_env_file:wrap_jsbsim'
    max_episode_steps=1500, # Used by TimeLimit wrapper if not applied manually
    # Additional arguments for wrap_jsbsim can be passed via kwargs in gym.make
    # or set as defaults here:
    # kwargs={'root': '.'} # Example
)

# Example of how to register and use
if __name__ == "__main__":
    from time import sleep
    print("Creating and testing F16ILSEnv environment...")

    # Register the environment with Gymnasium
    # This allows gym.make("F16ILSEnv-v0") to create the environment.
    gym.register(
        id="F16ILSEnv-v0",
        entry_point="f16_ils_env:F16ILSEnv", # Assumes this file is named f16_ils_env.py
        max_episode_steps=1500, # Default max_episode_steps from the env constructor
        # You can pass default kwargs for F16ILSEnv constructor here if needed,
        # e.g., kwargs={'runway_lat_deg': 34.95, ...}
    )
    print("Registered F16ILSEnv-v0")

    try:
        # Now you can make the environment using its ID
        env = gym.make("F16ILSEnv-v0")
        # For human mode:
        # env = gym.make("F16ILSEnv-v0", jsbsim_root='.', render_mode='human')


        print(f"Observation Space: {env.observation_space.shape}, {env.observation_space.dtype}")
        print(f"Action Space: {env.action_space.shape}, {env.action_space.dtype}")

        for episode in range(1): # Reduced episodes for quick test
            print(f"\n--- Episode {episode + 1} ---")
            obs, info = env.reset(seed=42 + episode)
            print(f"Initial observation stack shape: {obs.shape}")
            print(f"Initial info: {info}")
            terminated = False
            truncated = False
            total_reward = 0
            ep_steps = 0

            # Test with a simple action (try to fly somewhat level)
            # Action: [aileron, elevator, rudder, throttle]
            test_action = np.array([0.0, -0.01, 0.0, 0.45], dtype=np.float32)

            for i in range(env.spec.max_episode_steps + 50 if env.spec else 1550):
                # action = env.action_space.sample()
                action = test_action # Use a fixed action for more predictable testing initially

                obs, reward, terminated, truncated, info_step = env.step(action)
                total_reward += reward
                ep_steps += 1

                if ep_steps % 100 == 0 or terminated or truncated:
                    print(f"Step {ep_steps}, Reward: {reward:.3f}, TotalRew: {total_reward:.3f}, Term: {terminated}, Trunc: {truncated}")
                    # print(f" Obs sample (last frame): {obs[-1]}") # Print last frame of stack
                    if 'failure_reason' in info_step: print(f" Info: {info_step}")


                if env.render_mode == 'human':
                     env.render()
                     sleep(1/30)
                elif env.render_mode == 'rgb_array' and ep_steps % 60 == 0 :
                     frame = env.render()
                     if frame is not None:
                        print(f"Rendered RGB frame shape: {frame.shape}")
                     else:
                        print("Rendered RGB frame is None")


                if terminated or truncated:
                    print(f"Episode finished after {ep_steps} timesteps. Total Reward: {total_reward:.2f}")
                    if info_step.get('success', False): print("Landed Successfully!")
                    info = info_step # Keep last info
                    break
            print(f"Final Info for Episode {episode+1}: {info}")


    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals() and hasattr(env, 'close'):
            env.close()
            print("Environment closed.")