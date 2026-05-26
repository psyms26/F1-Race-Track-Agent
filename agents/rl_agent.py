# rl_agent.py
import os
import numpy as np
import config
from environment.race import Action
from environment.weather import DRY, DRYING, WET
from environment.tyre import compound_type


class RLAgent():

    name = "RL"

    def __init__(self, weather_forecast: str, model_path: str = None):

        from stable_baselines3 import PPO

        if model_path is None:
            model_path = os.path.join("models", "ppo_final.zip")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"PPO model not found at {model_path}. "
                f"Train one first via `python -m training.training_ppo`."
            )

        self.model = PPO.load(model_path)
        self.model_path = model_path

        if weather_forecast == WET:
            self.starting_tyre = "Wet"
        else:
            self.starting_tyre = "Medium"


    def select_action(self, race) -> Action:
        # RL learns ONLY steering and throttle (racing line + brake/accel)
        obs = self._build_observation(race)
        action, _ = self.model.predict(obs, deterministic=True)
        steering = config.RL_STEERING_BINS[int(action[0])]
        throttle = config.RL_THROTTLE_BINS[int(action[1])]

        fuel_left = race.fuel.current
        laps_left = max(1, config.TOTAL_LAPS - race.lap + 1)
        safe_budget = (fuel_left - 3.6) / laps_left
        if safe_budget < 2.0:
            throttle = min(throttle, 0.6)

        # Pit + compound delegated to utility-style strategic layer
        pit, compound = self._strategy_decision(race)

        return Action(
            steering=steering, throttle=throttle, pit=pit, compound=compound,
        )


    #  Utility-style strategic layer

    def _strategy_decision(self, race) -> tuple:
        current = race.current_condition
        current_comp = race.tyre.compound
        ctype = compound_type(current_comp)
        wear = race.tyre.wear_fraction
        laps_remaining = max(0, config.TOTAL_LAPS - race.lap)
        already_pitted = race.pit_stops >= config.MIN_PIT_STOPS

        # 1. Regulatory compliance — wrong tyre type for current weather
        if current == WET and ctype == "Slicks":
            return (1, "Wet")
        if current == DRY and ctype == "Wets":
            return (1, self._dry_compound_for_stint(laps_remaining, current_comp))

        # 2. Three-lap weather anticipation
        future_wet = self._forecast_wet_in_n_laps(race, n=3)
        if future_wet and ctype == "Slicks":
            return (1, "Wet")
        if (not future_wet) and ctype == "Wets" and current != WET:
            return (1, self._dry_compound_for_stint(laps_remaining, current_comp))

        # 3. Already pitted minimum times → don't re-pit unless mandated above
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


    def _forecast_wet_in_n_laps(self, race, n: int = 3) -> bool:

        target_lap = race.lap + n
        if target_lap > config.TOTAL_LAPS:
            return False
        if hasattr(race.weather, "condition_at_lap"):
            future = race.weather.condition_at_lap(target_lap)
            return future == WET
        # Fallback: use current condition as proxy
        return race.current_condition == WET


    #  Observation

    def _build_observation(self, race) -> np.ndarray:

        track = race.track
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

        return np.array(
            [speed_norm, lap_fraction, s_fraction]
            + curvatures
            + [dist_left, dist_right]
            + [fuel_fraction, wear_fraction]
            + weather_oh
            + compound_oh,
            dtype=np.float32,
        )


# Sanity Check

if __name__ == "__main__":
    from environment.track import Track
    from environment.weather import Weather
    from environment.race import Race

    SCENARIO = "FullDry"
    SEED = 42

    print(f"Running RL agent on {SCENARIO} race…\n")

    track = Track()
    weather = Weather(SCENARIO, seed=SEED)

    try:
        agent = RLAgent(weather_forecast=weather.forecast_starting_condition())
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Cannot run sanity check until PPO is trained.")
        exit(1)

    race = Race(track, weather, starting_compound=agent.starting_tyre)

    print(f"Agent: {agent.name}")
    print(f"Model: {agent.model_path}")
    print(f"Starting tyre: {agent.starting_tyre}")
    print()

    max_steps = 200_000
    step_count = 0
    while not race.dnf and not race.finished and step_count < max_steps:
        action = agent.select_action(race)
        race.step(action)
        step_count += 1
        if step_count % 10_000 == 0:
            print(f"  step {step_count:6d}: {race}")

    print(f"\nFinal state: {race}")
    print(f"Result: {'DNF (' + str(race.dnf_reason) + ')' if race.dnf else 'FINISHED'}")
    print(f"Total race time: {race.elapsed_time:.2f} s "
          f"({race.elapsed_time / 60:.1f} min)")
    print(f"Laps completed: {len(race.lap_times)}")
    if race.lap_times:
        print(f"Mean lap time:  {sum(race.lap_times) / len(race.lap_times):.2f} s")
        print(f"Best lap:       {min(race.lap_times):.2f} s")
        print(f"Worst lap:      {max(race.lap_times):.2f} s")
    print(f"Pit stops: {race.pit_stops}, compounds: {sorted(race.compounds_used)}")
    print(f"Off-track count: {race.offtrack_count}")
    print(f"Fuel remaining: {race.fuel.current:.2f} kg")
    print(f"Final tyre wear: {race.tyre.wear:.1f}%")
