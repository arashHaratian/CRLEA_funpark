import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("MountainCar-v0", n_envs=4)

model = A2C(policy="MlpPolicy",
    env=vec_env,
    seed=0,
    n_steps=64,
    gae_lambda=0.9,
    max_grad_norm =5,
    vf_coef=0.19,
    use_sde=True, 
    verbose=1)
model.learn(total_timesteps=500000)
# model.save("MountainCar_model")
# del model # remove to demonstrate saving and loading
# model = A2C.load("MountainCar_model")

obs = vec_env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")