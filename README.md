# Directional Atacom

## Setup
Clone the project repository and the dependecies
```python
git clone https://github.com/paolo-magliano/directional_atacom.git
git clone --branch qualifying --single-branch https://github.com/AirHockeyChallenge/air_hockey_challenge.git

```

Install the requirements
```
cd air_hockey_challenge
pip install -e .
cd ../directional_atacom
pip install -r requirement.txt

```

## Usage
To train D-ATACOM on the Planar Air Hockey environment run:
```python
cd cremini_rl
python run.py
```
To use different environments or algorithms, modify the `run.py` file. 
The package cremini_rl is based on the [mushroom_rl](https://github.com/MushroomRL/mushroom-rl) framework and contains the implementation of D-ATACOM as well as several Safe RL baselines. 
Currently `D-ATACOM`, `LagSAC`, `WCSAC`, `SafeLayerTD3`, `CBF-SAC`, `ATACOM`, `IQN-ATACOM` are implemented.

## Adding new Environments
To run the algorithms on a new environment add it to the `build_mdp` function in `cremini_rl/experiment.py`. 
The environment should be a subclass of `mushroom_rl.core.Environment`. The environment `cremini_rl\envs\goal_navigation_env.py` is an example of a environment wrapper for safety gymnasium.

For `D-ATACOM`, `IQN-ATACOM`, `CBF-SAC` the dynamics of the agent are also required. They should be a subclass of `cremini_rl.dynamics.dynamics.ControlAffineSystem`.  


