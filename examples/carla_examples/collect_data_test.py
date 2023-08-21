import gymnasium as gym
import carla_env

from wrappers.collector import DataCollector

env = gym.make('CarlaEnv-pixel-v1')

# collect data for 200 steps
env = DataCollector(env, steps=200, save_dir='./output', load_dir = None)
# env = DataCollector(env, steps=200, save_dir='./output', load_dir='./output/dataset_200.pkl')

env.reset()
done = False
while not done:
    next_obs, reward, done, _, info = env.step([1, 0])
env.close()
