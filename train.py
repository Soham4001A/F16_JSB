import gym
import jsbsim_gym.jsbsim_gym # This line makes sure the environment is registered
from os import path
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC
import globals

policy_kwargs = dict(
    features_extractor_class=JSBSimFeatureExtractor
)

if globals.NAVIGATE:
    env = gym.make("JSBSim-v0")
elif globals.TANKER:
    env = gym.make("JSBSimTank-v0")

log_path = path.join(path.abspath(path.dirname(__file__)), 'logs')

try:
    model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=log_path, gradient_steps=-1, device='cuda')
    model.learn(3000000)
finally:
    if model is not None:
        model.save("models/jsbsim_sac")
        model.save_replay_buffer("models/jsbsim_sac_buffer")