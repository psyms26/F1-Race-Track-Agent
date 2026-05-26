# training_ppo.py:
import argparse
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import config
from environment.gym_env import F1RaceEnv
from stable_baselines3.common.monitor import Monitor

def make_env(scenario=None, seed=None, rank=0):
    def _init():
        env_seed = (seed + rank) if seed is not None else None
        env = F1RaceEnv(weather_scenario=scenario, seed=env_seed)
        env = Monitor(env)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--phase1-steps", type=int,
                        default=config.TRAIN_TIMESTEPS_PHASE1,
                        help="Phase 1 (FullDry) timesteps")
    parser.add_argument("--phase2-steps", type=int,
                        default=config.TRAIN_TIMESTEPS_PHASE2,
                        help="Phase 2 (mixed weather) timesteps")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick 200k-step phase 1 only, skip phase 2")
    parser.add_argument("--n-envs", type=int,
                        default=config.TRAIN_NUM_PARALLEL_ENVS,
                        help="Number of parallel envs")
    parser.add_argument("--seed", type=int, default=config.BASE_SEED)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.smoke_test:
        args.phase1_steps = 200_000
        args.phase2_steps = 0

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/ppo_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Phase 1

    print("=" * 70)
    print(f"PHASE 1 — FullDry curriculum")
    print(f"  Timesteps:     {args.phase1_steps:,}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  TB logs:       {log_dir}")
    print("=" * 70)

    env_fns = [
        make_env(scenario="FullDry", seed=args.seed, rank=i)
        for i in range(args.n_envs)
    ]

    train_env = SubprocVecEnv(env_fns)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=config.PPO_LEARNING_RATE,
            n_steps=config.PPO_N_STEPS,
            batch_size=config.PPO_BATCH_SIZE,
            n_epochs=config.PPO_N_EPOCHS,
            gamma=config.PPO_GAMMA,
            gae_lambda=config.PPO_GAE_LAMBDA,
            clip_range=config.PPO_CLIP_RANGE,
            ent_coef=config.PPO_ENT_COEF,
            vf_coef=config.PPO_VF_COEF,
            max_grad_norm=config.PPO_MAX_GRAD_NORM,
            tensorboard_log=log_dir,
            verbose=1,
            seed=args.seed,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path="checkpoints",
        name_prefix="ppo_phase1",
    )

    model.learn(
        total_timesteps = args.phase1_steps,
        callback = checkpoint_cb,
        reset_num_timesteps=False,
        progress_bar = True,
    )

    phase1_path = "models/ppo_phase1.zip"
    model.save(phase1_path)
    print(f"\nPhase 1 complete — saved to {phase1_path}")
    train_env.close()

    # Phase 2: Mixed Weather

    if args.phase2_steps <= 0:
        print("\nSkipping phase 2 (smoke test or zero steps).")
    else:
        print("\n" + "=" * 70)
        print(f"PHASE 2 — Mixed weather curriculum")
        print(f"  Timesteps: {args.phase2_steps:,}")
        print(f"  Scenarios: {config.WEATHER_SCENARIOS} (randomized per episode)")
        print("=" * 70)

        env_fns = [
            make_env(scenario=None, seed=args.seed + 10_000, rank=i)
            for i in range(args.n_envs)
        ]

        train_env = SubprocVecEnv(env_fns)
        model = PPO.load(phase1_path, env=train_env)
        model.learning_rate = 1e-4

        checkpoint_cb = CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path="checkpoints",
            name_prefix="ppo_phase2",
        )

        model.learn(
            total_timesteps=args.phase2_steps,
            callback=checkpoint_cb,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        train_env.close()

    # Save Final Model

    final_path = "models/ppo_final.zip"
    model.save(final_path)
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE — final model: {final_path}")
    print(f"View training logs:  tensorboard --logdir {log_dir}")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()