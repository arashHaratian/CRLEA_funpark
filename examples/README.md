# List of Environments

To get the list of available environments, first import `gymnasium` and the package of your interest, and then use `gymnasium.envs.registry.keys()`.


## `airsim_rl` Available Environments:
The AirSim environemts are available by `import airgym`.

The environmments' IDs are:

- `'airsim-drone-sample-v0'` : The simulator with drone as the agent
- `'airsim-car-sample-v0'` : The simulator with car as the agent



## `carla_rl` Available Environments:
The CARLA environemts are available by `import carla_env`.

The environmments' IDs are:

- `'CarlaEnv-state-v1'` : Observations are the states (the informations on the screen)
- `'CarlaEnv-pixel-v1'` : Observations are the pixel values



## `sc2_rl` Available Environments:
The SC2 environemts are available by `import gym_pysc2`.

The environmments' IDs are:

- `'CSC2MoveToBeacon-v0'`
- `'SC2CollectMineralShards-v0'`
- `'SC2FindAndDefeatZerglings-v0'`
- `'SC2DefeatRoaches-v0'`
- `'SC2DefeatZerglingsAndBanelings-v0'`
- `'SC2CollectMineralsAndGas-v0'`
- `'SC2BuildMarines-v0'`

