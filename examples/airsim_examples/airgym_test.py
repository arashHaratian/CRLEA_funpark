import gymnasium as gym
import airgym

if __name__ == '__main__':
    env = gym.make('airsim-drone-sample-v0', ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(84, 84, 1))
    env.reset()
    done = False
    while not done:
        next_obs, reward, done, info = env.step([1, 0])
    env.close()
