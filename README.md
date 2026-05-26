# F1 Race Strategy: Comparing Agent Architectures

## Overview

This project compares three different intelligent agent architectures on a simulated Formula 1 race to see how each one handles race strategy under changing weather conditions. The environment is a custom built physics simulator (no external game engine) modelling a single car on a fixed Silverstone style track over 52 laps, with dynamic weather, tyre wear, fuel consumption, and grip all affecting how the car behaves.

The core question is simple: when the strategic decisions matter (when to pit, which tyre to fit, how hard to push), which kind of agent makes the best calls?

## Research Question

How do reactive, deliberative, and reinforcement learning agent architectures compare on race performance and robustness across stable and dynamic weather scenarios?

Three sub-questions follow from this:

1. Does a utility-based agent that anticipates weather and reasons about stint length beat a purely reactive rule-based agent when conditions change mid race?
2. Can a PPO reinforcement learning agent, given the same physics and action space, recover the performance of the hand coded agents without being told the strategy explicitly?
3. What trade offs in time, fuel margin, and corner precision does each architecture make?

## The Three Agents

**Rule-Based (reactive):** Follows the racing line with a bang bang throttle controller. Pits on fixed rules (lap 26, or tyre wear above 70%, or a regulatory compound mismatch). Doesn't look ahead or reason about strategy.

**Utility-Based (deliberative):** Shares the same driving controller but replaces the pit logic with a four stage utility calculation: regulatory compliance, a three lap weather lookahead, a two compound rule fallback, and a marginal utility test that weighs the time saved by pitting against the pit cost. Compound choice is stint length aware.

**Reinforcement Learning (PPO):** Learns steering and throttle through trial and error using Proximal Policy Optimization (Stable-Baselines3). Trained with a two phase curriculum (4M steps dry, then 2M steps across all weather). Pit and compound decisions are delegated to a utility style controller, so the RL question is isolated to closed loop driving and resource management.

## Environment

A custom simulator written from scratch in Python, organised into modules for track geometry, vehicle physics, tyre wear, fuel, weather scheduling, and the main race loop. It also exposes a Gym compatible interface for training the RL agent. Five weather scenarios are modelled: FullDry, FullWet, DryToWet, WetToDry, and Mixed.

## Experiments

The agents were compared across 225 races (3 agents × 5 weather scenarios × 15 seeds). Every race used a fixed seed so the weather schedule was identical across agents, giving a properly paired design. Results were analysed with Wilcoxon signed rank tests under Bonferroni correction, with effect sizes reported as Cohen's d and rank biserial correlation.

Metrics recorded per race: total race time, mean lap time, off track excursions, fuel remaining at finish, and tyre compounds used.

## Key Findings

- **All 225 races finished** (100% completion rate).
- The **utility agent significantly beats the rule-based agent in dynamic weather** (largest gap in Mixed, d = 1.36), but the two are tied in stable conditions, strategic reasoning only pays off when conditions actually change.
- The **RL agent finishes 15-21 minutes slower** but with the largest fuel margin and a perfect finish rate, a textbook Safe RL trade off caused by a fuel management safety clamp that caps its speed.
- The off track excursions show the RL agent's robustness breaks down in the Mixed scenario, the conditions least similar to its training distribution.

## Running It

```bash
# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install numpy scipy matplotlib pandas seaborn gymnasium stable-baselines3 torch

# train the RL agent (optional, a trained model is already included)
python -m training.training_ppo

# run all experiments
python -m experiments.run_experiments

# generate plots and statistics
python -m analysis.analyse_results
```

## Project Structure

```
environment/    track, weather, tyre, fuel, race, gym_env
agents/         rule_based, utility, rl_agent
training/       PPO training driver
experiments/    experimental harness
analysis/       statistics and plotting
models/         trained PPO model
results/        race data (CSV) and plots
```
