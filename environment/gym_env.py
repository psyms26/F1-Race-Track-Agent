# gym_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config
from environment.race import Action, Race
from environment.weather import Weather, DRY, DRYING, WET
from environment.track import Track
from environment.tyre import compound_type


class F1RaceEnv(gym.Env):

    metadata = {'render.modes': []}

    def __init__(self, weather_scenario: str = None, seed: int = None):

        super().__init__()


        self.action_space = spaces.MultiDiscrete([
            len(config.RL_STEERING_BINS),
            len(config.RL_THROTTLE_BINS),
        ])

        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(config.RL_STATE_DIM,),
            dtype=np.float32,
        )

        self._fixed_scenario = weather_scenario
        self._fixed_seed = seed
        self._max_steps = 250_000

        self.track: Track = None
        self.weather: Weather = None
        self.race: Race = None
        self._step_count = 0


    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)

        if self._fixed_scenario is not None:
            scenario = self._fixed_scenario
        else:
            idx = int(self.np_random.integers(0, len(config.WEATHER_SCENARIOS)))
            scenario = config.WEATHER_SCENARIOS[idx]

        if self._fixed_seed is not None:
            episode_seed = self._fixed_seed
        else:
            episode_seed = int(self.np_random.integers(0, 2 ** 31 - 1))


        self.track = Track()
        self.weather = Weather(scenario, seed=episode_seed)

        forecast = self.weather.forecast_starting_condition()
        starting_compound = "Wet" if forecast == WET else "Medium"

        self.race = Race(
            self.track, self.weather, starting_compound=starting_compound
        )
        self._step_count = 0

        return self._build_observation(), self._build_info()

    def step(self, action):

        steering = config.RL_STEERING_BINS[int(action[0])]
        throttle = config.RL_THROTTLE_BINS[int(action[1])]

        fuel_left = self.race.fuel.current
        laps_left = max(1, config.TOTAL_LAPS - self.race.lap + 1)
        safe_budget = (fuel_left - 3.6) / laps_left
        if safe_budget < 2.0:
            throttle = min(throttle, 0.6)


        pit, compound = self._strategy_decision()

        race_action = Action(
            steering=steering, throttle=throttle, pit=pit, compound=compound,
        )

        prev_lap = self.race.lap
        prev_wear = self.race.tyre.wear
        prev_s = self.race.s
        prev_elapsed_time = self.race.elapsed_time
        prev_fuel = self.race.fuel.current

        self.race.step(race_action)
        self._step_count += 1

        terminated = bool(self.race.finished or self.race.dnf)
        truncated = self._step_count >= self._max_steps

        reward = self._compute_reward(
            prev_lap, prev_wear, prev_s, prev_elapsed_time, prev_fuel, truncated,
        )

        return (
            self._build_observation(),
            reward,
            terminated,
            truncated,
            self._build_info(),
        )

    # Utility-style strategic layer


    def _strategy_decision(self) -> tuple:
        race = self.race
        current = race.current_condition
        current_comp = race.tyre.compound
        ctype = compound_type(current_comp)
        wear = race.tyre.wear_fraction
        laps_remaining = max(0, config.TOTAL_LAPS - race.lap)
        already_pitted = race.pit_stops >= config.MIN_PIT_STOPS

        # 1. Regulatory compliance
        if current == WET and ctype == "Slicks":
            return (1, "Wet")
        if current == DRY and ctype == "Wets":
            return (1, self._dry_compound_for_stint(laps_remaining, current_comp))

        # 2. Three-lap weather anticipation
        future_wet = self._forecast_wet_in_n_laps(n=3)
        if future_wet and ctype == "Slicks":
            return (1, "Wet")
        if (not future_wet) and ctype == "Wets" and current != WET:
            return (1, self._dry_compound_for_stint(laps_remaining, current_comp))

        # 3. Already pitted minimum times → don't re-pit unless mandated
        if already_pitted:
            return (0, current_comp)

        # 4. Mandatory-pit fallback near end of race
        if laps_remaining <= 3:
            if current == WET:
                return (1, "Wet")
            return (1, self._dry_compound_for_stint(laps_remaining, current_comp))

        # 5. Utility formula: pit if wear cost > pit cost
        threshold = config.UTILITY_PIT_TIME_COST + 17.0
        if wear * 4.0 * laps_remaining > threshold:
            if current == WET:
                return (1, "Wet")
            return (1, self._dry_compound_for_stint(laps_remaining, current_comp))

        return (0, current_comp)

    def _dry_compound_for_stint(self, laps_remaining: int, current_compound: str = None) -> str:
        if laps_remaining <= 16:
            candidate = "Soft"
        elif laps_remaining <= 25:
            candidate = "Medium"
        else:
            candidate = "Hard"
        if candidate == current_compound:
            candidate = "Hard" if current_compound != "Hard" else "Medium"
        return candidate

    def _forecast_wet_in_n_laps(self, n: int = 3) -> bool:
        target_lap = self.race.lap + n
        if target_lap > config.TOTAL_LAPS:
            return False
        if hasattr(self.weather, "condition_at_lap"):
            future = self.weather.condition_at_lap(target_lap)
            return future == WET
        return self.race.current_condition == WET

    # Reward

    def _compute_reward(self, prev_lap, prev_wear, prev_s,
                        prev_elapsed_time, prev_fuel, truncated) -> float:
        race = self.race


        delta_time = max(0.0, race.elapsed_time - prev_elapsed_time)
        reward = config.REWARD_TIME_COST_PER_SECOND * delta_time

        # SHAPING: forward progress when on-track
        half_width = config.TRACK_WIDTH / 2.0
        is_off_track = abs(race.lateral) > half_width

        if not is_off_track:
            if race.lap > prev_lap:
                delta_s = (config.TRACK_LENGTH - prev_s) + race.s
            else:
                delta_s = race.s - prev_s
            delta_s = max(0.0, delta_s)
            reward += delta_s * config.REWARD_PROGRESS_PER_METER
        else:
            reward += config.REWARD_OFFTRACK_PER_SEC * config.SIM_TIMESTEP

        # Lap-completion shaping
        if race.lap > prev_lap:
            reward += config.REWARD_LAP_BASE

        # Mild tyre-wear shaping
        delta_wear = max(0.0, race.tyre.wear - prev_wear)
        reward += config.REWARD_WEAR_PER_PERCENT * delta_wear

        # Per-step fuel-use signal — direct cost for burning fuel,
        # forces the agent to trade speed against fuel budget every step.
        delta_fuel = max(0.0, prev_fuel - race.fuel.current)
        reward += config.REWARD_FUEL_USED_PER_KG * delta_fuel

        # TERMINAL
        if race.dnf:
            reward += config.REWARD_DNF
            if race.dnf_reason == "out_of_fuel":
                reward += config.REWARD_OUT_OF_FUEL_EXTRA
        elif race.finished:
            reward += config.REWARD_FINISH_BASE
            reward += race.fuel.current * config.REWARD_FUEL_REMAINING_PER_KG
        elif truncated:
            reward += config.REWARD_DNF

        return float(reward)

    def _build_observation(self) -> np.ndarray:

        track = self.track
        race = self.race

        speed_norm = race.speed / config.MAX_SPEED
        lap_fraction = race.lap / config.TOTAL_LAPS
        s_fraction = race.s / config.TRACK_LENGTH

        curvatures = []
        speed = max(race.speed, 1.0)
        for t in config.RL_LOOKAHEAD_TIMES:
            future_s = (race.s + speed * t) % config.TRACK_LENGTH
            kappa = track.curvature_at(future_s)

            curvatures.append(kappa * 20.0)

        half_width = config.TRACK_WIDTH / 2.0
        dist_left = (half_width + race.lateral) / half_width
        dist_right = (half_width - race.lateral) / half_width
        dist_left = float(np.clip(dist_left, 0.0, 2.0))
        dist_right = float(np.clip(dist_right, 0.0, 2.0))

        fuel_fraction = race.fuel.current / config.FUEL_START
        wear_fraction = race.tyre.wear / config.MAX_TYRE_WEAR

        cond = race.current_condition
        weather_oh = [
            1.0 if cond == DRY else 0.0,
            1.0 if cond == DRYING else 0.0,
            1.0 if cond == WET else 0.0,
        ]

        compound = race.tyre.compound
        compound_oh = [
            1.0 if compound == "Soft" else 0.0,
            1.0 if compound == "Medium" else 0.0,
            1.0 if compound == "Hard" else 0.0,
            1.0 if compound == "Wet" else 0.0,
        ]

        obs = np.array(
            [speed_norm, lap_fraction, s_fraction]
            + curvatures
            + [dist_left, dist_right]
            + [fuel_fraction, wear_fraction]
            + weather_oh
            + compound_oh,
            dtype=np.float32,
        )

        assert obs.shape == (config.RL_STATE_DIM,), (
            f"Observation has wrong shape {obs.shape}, "
            f"expected ({config.RL_STATE_DIM},)"
        )
        return obs


    def _build_info(self) -> dict:
        race = self.race

        return{
            "lap": race.lap,
            "s": race.s,
            "speed": race.speed,
            "lateral": race.lateral,
            "fuel": race.fuel.current,
            "tyre_wear": race.tyre.wear,
            "compound": race.tyre.compound,
            "weather": race.current_condition,
            "pit_stops": race.pit_stops,
            "compounds_used": list(race.compounds_used),
            "elapsed_time": race.elapsed_time,
            "offtrack_count": race.offtrack_count,
            "dnf": race.dnf,
            "dnf_reason": getattr(race, "dnf_reason", None),
            "finished": race.finished,
        }

# Sanity Check
if __name__ == "__main__":
    print("Sanity-checking F1RaceEnv on FullDry...\n")

    env = F1RaceEnv(weather_scenario="FullDry", seed=42)
    obs, info = env.reset()

    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial obs:       {obs}")
    print()

    total_reward = 0.0
    for step in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode ended after {step + 1} steps")
            print(f"  finished:   {info['finished']}")
            print(f"  dnf:        {info['dnf']}")
            print(f"  dnf_reason: {info['dnf_reason']}")
            print(f"  lap:        {info['lap']}")
            print(f"  pit_stops:  {info['pit_stops']}")
            print(f"  compounds:  {info['compounds_used']}")
            print(f"  total_reward: {total_reward:.2f}")
            break
    else:
        print(f"Ran 2000 steps without termination")
        print(f"  Lap reached: {info['lap']}, total_reward: {total_reward:.2f}")

    print("\nSanity check passed.")

