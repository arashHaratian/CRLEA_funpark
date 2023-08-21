##########################
# NOT GUARANTEE TO LEARN #
##########################


import gymnasium as gym
import carla_env
from stable_baselines3 import A2C

# env = gym.make('CarlaEnv-pixel-v1', observations_type = 'state')
env = gym.make('CarlaEnv-pixel-v1', render = False)
env.reset()
model = A2C("CnnPolicy", env, verbose=1, policy_kwargs=dict(normalize_images=False))
model.learn(total_timesteps=25000)
