# utility.py
import config
from environment.race import Action
from environment.weather import DRY, DRYING, WET
from environment.tyre import compound_type

class UtilityAgent():

    name = "Utility"

    def __init__(self, weather_forecast: str):

        if weather_forecast == WET:
            self.starting_tyre = config.RULE_BASED_WET_TYRE
        else:
            self.starting_tyre = config.RULE_BASED_DRY_START_TYRE


    def select_action(self, race) -> Action:

        return Action(
            steering = self._steering(race),
            throttle = self._throttle(race),
            pit = self._pit_decision(race),
            compound = self._compound_choice(race),
        )

    def _steering(self, race) -> float:
        lookahead_time = 1.0
        speed = max(race.speed, 1.0)
        lookahead_s = race.s + speed * lookahead_time
        target_lateral = race.track.racing_line_lateral(lookahead_s)
        required_v = (target_lateral - race.lateral) / lookahead_time
        max_v = config.MAX_STEERING_ANGLE * 10.0
        steering = required_v / max_v
        return max(-1.0, min(1.0, steering))

    def _throttle(self, race) -> float:

        current_safe = race.max_safe_speed

        lookahead_time = 2.0
        speed = max(race.speed, 1.0)
        lookahead_s = race.s + speed * lookahead_time
        future_safe = race.track.max_safe_speed(lookahead_s, race.grip)

        achievable = min(
            current_safe,
            future_safe,
            config.MAX_SPEED * race.tyre.speed_factor,
        )

        grip_factor = (
                1.0 - (1.0 - race.grip) * config.UTILITY_THROTTLE_GRIP_SCALING
        )
        # Wear factor: pulls target down as tyre wears
        wear_factor = (
                1.0 - race.tyre.wear_fraction * config.UTILITY_WEAR_SCALING
        )

        effective_safe = achievable * grip_factor * wear_factor * 0.95
        speed_delta = race.speed - effective_safe

        if speed_delta < -2.0:
            return 1.0
        elif speed_delta > 2.0:
            return -1.0
        elif speed_delta < 0:
            return 0.5
        else:
            return 0.0

    def _pit_decision(self, race) -> int:
        if race.tyre.type == "Slicks" and race.current_condition == WET:
            return 1
        if race.tyre.type == "Wets" and race.current_condition == DRY:
            return 1

        future_lap = race.lap + 3
        if future_lap <= config.TOTAL_LAPS:
            future = race.weather.condition_at_lap(future_lap)
            if race.tyre.type == "Slicks" and future == WET:
                return 1  # pit BEFORE the rain hits, not after
            if race.tyre.type == "Wets" and future == DRY:
                return 1  # switch off wets BEFORE the track dries

        remaining = max(0, config.TOTAL_LAPS - race.lap)

        if race.pit_stops == 0 and remaining <= 3:
            return 1

        if remaining < 4:
            return 0

        wear = race.tyre.wear_fraction
        projected_time_loss_per_lap = wear * 4.0
        expected_savings = projected_time_loss_per_lap * remaining

        # Pit if savings exceed the cost (plus small buffer for stability)
        if expected_savings > config.UTILITY_PIT_TIME_COST + 17.0:
            return 1

        return 0

    def _compound_choice(self, race) -> str:

        if race.current_condition == WET:
            return "Wet"

        future_lap = race.lap + 3
        if future_lap <= config.TOTAL_LAPS:
            future = race.weather.condition_at_lap(future_lap)
            if future == WET:
                return "Wet"


        remaining = max(0, config.TOTAL_LAPS - race.lap)
        if remaining <= 16:
            return "Soft"
        elif remaining <= 25:
            return "Medium"
        else:
            return "Hard"


# Sanity Check

if __name__ == "__main__":
    from environment.track import Track
    from environment.weather import Weather
    from environment.race import Race

    SCENARIO = "Mixed"   # change to FullWet / DryToWet / WetToDry / Mixed
    SEED = 42

    print(f"Running utility agent on {SCENARIO} race…\n")
    track = Track()
    weather = Weather(SCENARIO, seed=SEED)
    agent = UtilityAgent(weather_forecast=weather.forecast_starting_condition())
    race = Race(track, weather, starting_compound=agent.starting_tyre)

    print(f"Agent: {agent.name}")
    print(f"Starting tyre: {agent.starting_tyre}")
    print(f"Weather schedule: {weather.schedule_summary()}")
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
    print(f"Off-track count: {race.offtrack_count}, "
          f"penalty added: {race.offtrack_penalty_added:.1f} s")
    print(f"Fuel remaining: {race.fuel.current:.2f} kg "
          f"({race.fuel.fraction_remaining * 100:.1f}%)")
    print(f"Final tyre wear: {race.tyre.wear:.1f}%")