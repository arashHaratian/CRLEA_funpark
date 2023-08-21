import gymnasium as gym
import airgym

from wrappers.collector import DataCollector

if __name__ == '__main__':
    env = gym.make('airsim-drone-sample-v0')

    # collect data for 200 steps
    env = DataCollector(env, steps=200, save_dir='./output')
    # env = DataCollector(env, steps=200, save_dir='./output', load_dir='./output/dataset_200.pkl')

    env.reset()
    done = False
    while not done:
        next_obs, reward, done, info = env.step([1, 0])
    env.close()
