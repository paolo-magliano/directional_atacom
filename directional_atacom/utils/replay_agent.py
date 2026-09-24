import numpy as np
import torch

from directional_atacom.envs import *

from directional_atacom.algorithms import *

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import SAC

from directional_atacom.utils.ext_core import ExtCore
from directional_atacom.utils.safe_core import SafeCore

safe_agent = True
# path_to_agent = "../logs/air_hockey_vel_2026-09-20_20-25-06/0/agent-200.msh"
path_to_agent = "agent.msh"
if safe_agent:
    # agent = DatacomSAC.load(path_to_agent)
    agent = AtacomSAC.load(path_to_agent)
else:
    agent = SAC.load(path_to_agent)

# np.random.seed(18)
# torch.manual_seed(18)

# mdp = AirHockeyVel(return_cost=safe_agent, headless=False)
# mdp = QuadrotorEnv(return_cost=safe_agent)
mdp = PlanarAirHockeyVel(return_cost=safe_agent)

if safe_agent:
    core = SafeCore(agent, mdp)
else:
    core = ExtCore(agent, mdp)

core.evaluate(n_episodes=20, render=True)

