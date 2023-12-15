# CRLEA (FunPark_RL)

**Containerized Reinforcement Learning Environments and Algorithms**

This Repo provides a [Singularity file definition](https://docs.sylabs.io/guides/3.11/user-guide/index.html) to build a container and python virtual environments for various Reinforcement environments and algorithms. The environments uses [gymnasium](https://gymnasium.farama.org/) API to make it easy to interact, and algorithms are provided through [stable baselines3 Zoo](https://rl-baselines3-zoo.readthedocs.io/en/master/). The Python virtual environments are made with [conda](https://docs.conda.io/en/latest/).


## Available RL Environments and Algorithms:

All virtual environments contain `pytorch 2.0.1` for CUDA 11.8, and `stable_baselines3 zoo` installation.

The list of conda virtual environments and the RL environments that they provide is as follow:

> AT THE TIME OF WRITING  (*LATEST PYTHON*) IS 3.11.4

- **`base_rl`** : (*LATEST PYTHON*)
    - [Gymnasium's Classic](https://gymnasium.farama.org/environments/classic_control/) *environments* 
    - [Gymnasium's Box2D](https://gymnasium.farama.org/environments/box2d/) *environments* 
    - [Gymnasium's Toy Text](https://gymnasium.farama.org/environments/toy_text/) *environments* 
    - [Gymnasium's MuJoCo](https://gymnasium.farama.org/environments/mujoco/) *environments* 
    - [Gymnasium's Atari](https://gymnasium.farama.org/environments/atari/) *environments* 

  <p> <br> </p>

- **`mine_rl`**: (*LATEST PYTHON*)
    - [Minecraft / MineRL](https://minerl.readthedocs.io/en/latest/) *gymnasium environments*
    - \+ `base_rl` *environments* 

  <p> <br> </p>

- **`carla_rl`**: (*PYTHON 3.8.0*)
    - [CARLA](https://carla.org/) *gymnasium environments*
    - \+ `base_rl` *environments* 

  <p> <br> </p>

- **`airsim_rl`**: (*LATEST PYTHON*)
    - [AirSim ](https://microsoft.github.io/AirSim/) *gymnasium environments*
    - \+ `base_rl` *environments* 

  <p> <br> </p>


- **`cs2_rl`**: (*PYTHON 3.8.0*)
    - [PySC2](https://github.com/deepmind/pysc2) *gymnasium environments*
    - \+ `base_rl` *environments*

  <p> <br> </p>


- **`funpark_rl`**: (*PYTHON 3.9.17*)
    - [miniworld](https://miniworld.farama.org/) *environments*
    - [minigrid](https://minigrid.farama.org/) *environments*
    - [gymnasium-robotics](https://robotics.farama.org/) *environments*
    - [minari envs](https://minari.farama.org/) *environments*
    - [vizdoom](https://vizdoom.farama.org/) *environments*
    - [highway-env](https://highway-env.farama.org/) *environments*
    - [shimmy](https://shimmy.farama.org/) *environments*
    - [miniwob](https://miniwob.farama.org/) *environments*
    - [gym-microrts](https://github.com/Farama-Foundation/MicroRTS-Py) *environments*
    - \+ `base_rl` *environments* 
    - [pettingzoo](https://pettingzoo.farama.org/) *API*
    - [mo-gymnasium](https://pettingzoo.farama.org/) *API*
    - [supersuit wrappers](https://github.com/Farama-Foundation/SuperSuit) *wrappers*
 


## How to Use:

> NOTICE: THE SINGULARITY COMMANDS ARE FOR A ROOTLESS SINGULARITY INSTALLATION. CHANGE THE COMMANDS ACCORDINGLY IF YOU HAVE DIFFERENT SINGULARITY INSTALLATION. 

First you may want to download the repo files as zip or clone the repo and change directory on your terminal to the repo directory.

Now to build the container use the following command:
```bash
singularity build --fakeroot --sandbox CRLEA.sif CRLEA.def
```
Depending on your network connection speed , the building process may take quite time.

`--fakeroot` argument is for the rootless singularity. 

`--sandbox` is to make the container available as a folder in your directory so that you can access its folders. It is recommended to use this argument but it is optional.


After building the container, to use the shell interface of the container use:
```bash
singularity shell --nv -Cw --env DISPLAY=$DISPLAY CRLEA.sif
```

`--nv` makes GPU(s) available.

`-w` makes files writable in the container.

`--containall` or `-C` to make a dummy home directory and prevent container to mount your home directory of your **local machine**. You may use `--no-home` will prevent mounting your home directory in the container ([**if your home directory is your current working directory, it will mount the home anyways!**](https://docs.sylabs.io/guides/3.11/user-guide/bind_paths_and_mounts.html#using-no-home-and-containall-flags)).  `singularity shell --nv --no-home -w CRLEA.sif`

NOTICE that if you use `-C` argument you may want to set the `DISPLAY` variable to your `$DISPLAY` on your machine.


In the container you may have to use `. activate ENVNAME` instead of  `conda activate ENVNAME` to activate the virtual environment ***for the first time***. After that you can use `conda activate ENVNAME` normally.



## Notes about each environment:

### AirSim and Carla 

In case you are running the container with `-w` argument and you get an error regarding the low storage, you can change the value of `sessiondir max size` of the singularity in `singularity-ce/etc/singularity/singularity.conf`. This issue may occur especially if you want to run carla.

To use AirSim or CARLA gymnasium environments, you have to run them before trying to run your python script.

Use the following command in container to run AirSim:

```bash
sh /AirSim/AirSimNH/LinuxNoEditor/AirSimNH.sh
```
To use the following command in container to run CARLA

```bash
sh /carla/CarlaUE4.sh -fps 20 -nosound
```
`-fps 20` is used as the gymnasium wrapper uses the default value of 20 for env fps.
The carla wrapper is based on an old repo ([HERE](https://github.com/janwithb/carla-gym-wrapper)), and the AirSim wrapper is based on an old repo ([HERE](https://github.com/microsoft/AirSim/tree/main/PythonClient/reinforcement_learning)) that are updated from `gym` to `gymnasium`, and adapted the observations to be useable for sb3. Now, both wrappers can be used with latest version of the simulator.(e.g., from carla 0.9.9 to 0.9.14)

If you want to use CARLA 0.9.11 or older, you may want to change the code in `/gym_wrappers/carla_gym_wrapper/carla_env/carla_env.py`. (`BasicAgent` should become `RoamingAgent`)


One last thing is that AirSim has dependency to `msgpack-rpc-python` package that should be installed before installing the packages in `airsim_rl.yml`; therefore, it has been installed in the singularity definition.

### StarCraft

While using the `gym_pysc2` environments, you may want to mask the chosen action of your agent as all the available actions are not valid in all the environments. You can use `basic_action_mask` argument when you make the gym environment (if used it gives the agent a negative reward for each invalid action. it may not let the agent to learn, so better that you mask the actions or just simply change the amount of negative reward in `/gym_wrappers/pysc2_gym_wrapper/gym_pysc2/envs/pysc2env.py`).



The `gym_pysc2` wrapper is based on an old repo ([HERE](https://github.com/vwxyzjn/gym-pysc2)) that is modified and updated from `gym` to `gymnasium`, and adapted the observations to be useable for sb3.


### Minecraft and MicroRTS-Py

The `minerl` and `gym_microrts` packages have no gymnasium implementation; therefore, you CANNOT use stable baselines3 functions directly on the environments. You may want to check resources on the internet on how to create your RL algorithms to learn in `minerl` or `gym_microrts` environments.


## Contents of The Repo:
-  ***CRLEA.def*** : Singularity file definition

-  ***env_yml_files*** : YAML files for conda virtual environments

-  ***gym_wrappers*** : The custom gymnasium environment wrappers, contains:
    
    - **airsim_gym_wrapper** : The wrapper for AirSim
    - **carla_gym_wrapper** : The wrapper for Carla
    - **pysc2_gym_wrapper** : The wrapper for PySC2 and SC2  

- ***examples*** : Some simple examples and list of custom environments

## Contributions

## License



<!-- about having file locally or download!? -->

<!-- python /gym_wrappers/pysc2_gym_wrapper/test.py --gym-id SC2MoveToBeacon-v0     --num-envs 1     --num-steps 256     --cuda True  -->
