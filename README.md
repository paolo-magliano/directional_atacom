# Directional ATACOM

Official implementation of **Directional Constraints for Efficient Exploration in Safe Reinforcement Learning**
Paolo Magliano, Puze Liu, Jan Peters, Davide Tateo, Raffaello Camoriano — *IROS 2026*.

[Project page](https://research.robot-learning.net/atacom-dc/) · [arXiv](https://arxiv.org/abs/2607.12784) · [Video](https://www.youtube.com/watch?v=G56UBijyyus)

<video src="https://raw.githubusercontent.com/paolo-magliano/directional_atacom/iros2026/figs/atacom_dc_video.mp4" controls></video>

ATACOM keeps a policy inside the safe set by projecting every action onto the constraint manifold. Directional Constraints (DC) apply that projection **only to actions that push the system towards a boundary**: actions moving away from it are left untouched. Exploration stays free where it is harmless, which improves the safety–performance trade-off during learning.

## Setup

Python 3.10 is recommended. Clone the project and the Air Hockey Challenge, pinned to the commit used for the reported results:

```bash
git clone https://github.com/paolo-magliano/directional_atacom.git
git clone --branch qualifying https://github.com/AirHockeyChallenge/air_hockey_challenge.git
cd air_hockey_challenge && git checkout 4dc076384f457355918419c909edf65e160d4ed3
```

Install both packages:

```bash
pip install -e .                       # from air_hockey_challenge/
cd ../directional_atacom
pip install -r requirement.txt         # includes -e . for this package
```

## Usage

```bash
cd directional_atacom
python run.py
```

Choose the environment and the algorithm at the top of `run.py` (`env`, `alg`); `alg` selects the config files merged from `configs/`, where the `_dc` suffix enables Directional Constraints. Training curves are logged to Weights & Biases and to `logs/`; `utils/plot.py` reproduces the figures from the logged runs.

## Adding new environments

Add the environment to the `build_mdp` function in `directional_atacom/experiment.py`. It must subclass `mushroom_rl.core.Environment`, and ATACOM additionally needs the agent dynamics as a subclass of `directional_atacom.dynamics.dynamics.ControlAffineSystem`.

## Credits

Built on top of the [D-ATACOM implementation](https://github.com/cube1324/d-atacom) released with *Handling Long-Term Safety and Uncertainty in Safe Reinforcement Learning* (Günster, Liu, Peters, Tateo), and on the [MushroomRL](https://github.com/MushroomRL/mushroom-rl) framework. The robot air hockey environments come from the [Air Hockey Challenge](https://github.com/AirHockeyChallenge/air_hockey_challenge).

## Additional notes

The quadrotor and KUKA air hockey experiments reported in the paper were produced with [directional_atacom_kuka](https://github.com/paolo-magliano/directional_atacom_kuka). All those environments are available here as well, but results may differ slightly because of library and simulator version differences.

## License

Released under the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{magliano2026directional,
  title={Directional Constraints for Efficient Exploration in Safe Reinforcement Learning},
  author={Magliano, Paolo and Liu, Puze and Peters, Jan and Tateo, Davide and Camoriano, Raffaello},
  booktitle={2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1-8},
  year={2026},
  organization={IEEE}
}
```
