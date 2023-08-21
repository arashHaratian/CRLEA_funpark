from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

# Parallel environments
vec_env = VecFrameStack(make_vec_env("ALE/MsPacman-v5", n_envs=8), n_stack=4)

model = PPO("CnnPolicy", vec_env, n_steps = 128, ent_coef = 0.01, vf_coef = 0.5, learning_rate = 2.5e-4, normalize_advantage = False, verbose=1)
model.learn(total_timesteps=5000000)
## Save the model
# model.save("pacman_policy")
## Load the saved model
# model = A2C.load("pacman_policy")

obs = vec_env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")