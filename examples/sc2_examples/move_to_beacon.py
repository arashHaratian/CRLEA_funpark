import gymnasium as gym
import gym_pysc2
import sys
from absl import flags
from stable_baselines3 import PPO

FLAGS = flags.FLAGS
FLAGS([''])
# env = gym.make("SC2MoveToBeacon-v0", flatten_obs =False, only_raw_images = True, spatial_dim = 64, basic_action_mask = True, visualize = False)
env = gym.make("SC2MoveToBeacon-v0", flatten_obs =False, only_raw_images = False, spatial_dim = 64, basic_action_mask = True, visualize = True)
env.reset()
# model = PPO("MultiInputPolicy", env,  n_steps = 256, learning_rate = 0.00025, verbose=1)
model = PPO("MultiInputPolicy", env,  n_steps = 256, ent_coef = 0.01, vf_coef = 0.5, learning_rate = 2.5e-4, normalize_advantage = False, verbose=1)
model.learn(total_timesteps=500000)


env = gym.make("SC2MoveToBeacon-v0", flatten_obs =False, only_raw_images = False, spatial_dim = 64, basic_action_mask = True)
obs, _ = env.reset()
rew_sum=0
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, _, info = env.step(action)
    rew_sum += rewards
    env.render()
    if done:
        print(rew_sum)
        rew_sum = 0
    # if done:
    #     break