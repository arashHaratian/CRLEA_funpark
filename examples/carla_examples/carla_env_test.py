import gymnasium as gym
import carla_env

if __name__ == '__main__':
    env = gym.make('CarlaEnv-pixel-v1')
    env.reset()
    done = False
    while not done:
        next_obs, reward, done, _, info = env.step([1, 0])
    env.close()
