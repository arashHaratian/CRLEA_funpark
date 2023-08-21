import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO


# env = gym.make('MiniGrid-Empty-8x8-v0', render_mode="human")
env = gym.make('MiniGrid-Empty-8x8-v0')
env = RGBImgPartialObsWrapper(env)  # Get pixel observations
env = ImgObsWrapper(env)  # Get rid of the 'mission' field
env.reset()
model = PPO("CnnPolicy", env,  n_steps = 256, learning_rate = 0.00025, verbose=1)
model.learn(total_timesteps=500000)


env = gym.make('MiniGrid-Empty-8x8-v0', render_mode = "human")
env = RGBImgPartialObsWrapper(env)  # Get pixel observations
env = ImgObsWrapper(env)  # Get rid of the 'mission' field
obs = env.reset()[0]

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, _, info = env.step(action)
    env.render()
    if done:
        break

env.close()
