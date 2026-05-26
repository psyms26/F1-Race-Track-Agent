# rule_based.py
import config
from environment.race import Action
from environment.weather import DRY, DRYING, WET
from environment.tyre import compound_type

class RuleBasedAgent:

    name = "RuleBased"

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

        effective_safe = min(current_safe, future_safe, config.MAX_SPEED * race.tyre.speed_factor) * 0.95
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
        # REGULATORY COMPLIANCE: must match compound to weather

        if race.tyre.type == "Slicks" and race.current_condition == WET:
            return 1
        # On wets in dry → MUST pit (wets shred 2.5× faster in dry)
        if race.tyre.type == "Wets" and race.current_condition == DRY:
            return 1

        # Already pitted once → don't pit again (naive strategy)
        if race.pit_stops >= 1:
            return 0
        # Primary rule: fixed lap pit
        if race.lap >= config.RULE_BASED_PIT_LAP:
            return 1
        # Safety fallback: wear too high
        if race.tyre.wear >= config.RULE_BASED_WEAR_THRESHOLD:
            return 1
        return 0

    def _compound_choice(self, race) -> str:

        if race.current_condition == WET:
            return config.RULE_BASED_WET_TYRE
        return config.RULE_BASED_DRY_PIT_TYRE

# Sanity Check
if __name__ == "__main__":
    from environment.track import Track
    from environment.weather import Weather
    from environment.race import Race

    print("Running rule-based agent\n")
    track = Track()
    weather = Weather("Mixed", seed=42)
    agent = RuleBasedAgent(weather_forecast=weather.forecast_starting_condition())
    race = Race(track, weather, starting_compound=agent.starting_tyre)

    print(f"Agent: {agent.name}")
    print(f"Starting tyre: {agent.starting_tyre}")
    print(f"Pit strategy: lap {config.RULE_BASED_PIT_LAP} onto {config.RULE_BASED_DRY_PIT_TYRE}")
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