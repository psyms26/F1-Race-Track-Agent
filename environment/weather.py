import random
from dataclasses import dataclass
from typing import List, Optional
import config

DRY = "Dry"
WET = "Wet"
DRYING = "Drying"

ALL_CONDITIONS = [DRY, WET, DRYING]

@dataclass
class WeatherEvent:
    lap: int
    condition: str


class Weather:
    def __init__(self, scenario: str, seed: Optional[int] = None):
        if scenario not in config.WEATHER_SCENARIOS:
            raise ValueError(
                f"Unknown weather scenario: {scenario!r}. " 
                f"Must be one of {config.WEATHER_SCENARIOS}."
            )
        self.scenario: str = scenario
        self.rng = random.Random(seed)
        self.schedule: List[WeatherEvent] = self._build_schedule()
        self.current_condition: str = self.schedule[0].condition


    def _draw_transition_lap(self, low: int, high: int, exclude_near: Optional[int] = None, min_gap: int = 5,) -> int:
        for _ in range(50):
            candidate = self.rng.randint(low, high)
            if exclude_near is None or abs(candidate - exclude_near) >= min_gap:
                return candidate
        return min(high, max(low, (exclude_near or low) + min_gap))

    def _build_schedule(self) -> List[WeatherEvent]:
        events: List[WeatherEvent] = []
        lo = config.WEATHER_TRANSITION_LAP_MIN
        hi = config.WEATHER_TRANSITION_LAP_MAX
        dur = config.WEATHER_TRANSITION_DURATION_LAPS

        if self.scenario == "FullDry":
            events.append(WeatherEvent(lap=1, condition = DRY))

        elif self.scenario == "FullWet":
            events.append(WeatherEvent(lap=1, condition = WET))

        elif self.scenario == "DryToWet":
            t = self._draw_transition_lap(lo, hi)
            events.append(WeatherEvent(lap=1, condition = DRY))
            events.append(WeatherEvent(lap=t, condition = DRYING))
            events.append(WeatherEvent(lap=t + dur, condition = WET))

        elif self.scenario == "WetToDry":
            t = self._draw_transition_lap(lo, hi)
            events.append(WeatherEvent(lap=1, condition = WET))
            events.append(WeatherEvent(lap=t, condition = DRYING))
            events.append(WeatherEvent(lap=t + dur, condition = DRY))

        elif self.scenario == "Mixed":
            t1 = self._draw_transition_lap(lo, max(lo, hi - 2 * dur - 5))
            t2_min = t1 + dur + 5
            t2_max = config.TOTAL_LAPS - dur - 1
            t2 = self._draw_transition_lap(t2_min, max(t2_min, t2_max))
            events.append(WeatherEvent(lap=1, condition = DRY))
            events.append(WeatherEvent(lap=t1, condition = DRYING))
            events.append(WeatherEvent(lap=t1 + dur, condition = WET))
            events.append(WeatherEvent(lap=t2, condition = DRYING))
            events.append(WeatherEvent(lap=t2 + dur, condition = DRY))

        events = [e for e in events if e.lap <= config.TOTAL_LAPS]
        events.sort(key=lambda e: e.lap)
        return events

    def condition_at_lap(self, lap: int) -> str:
        active = self.schedule[0].condition
        for event in self.schedule:
            if event.lap <= lap:
                active = event.condition
            else:
                break
        return active

    def update(self, current_lap: int) -> str:
        self.current_condition = self.condition_at_lap(current_lap)
        return self.current_condition

    def is_wet_phase(self, lap: int) -> bool:
        return self.condition_at_lap(lap) == WET

    def has_any_wet_phase(self) -> bool:
        return any(e.condition == WET for e in self.schedule)


    def forecast_starting_condition(self) -> str:
        return self.schedule[0].condition


    def schedule_summary(self) -> str:
        return " | ".join(f"L{e.lap}:{e.condition}" for e in self.schedule)


# Sanity Check

if __name__ == "__main__":
    print("Weather scenario timelines (seed=42):")
    print("-" * 60)
    for scenario in config.WEATHER_SCENARIOS:
        w = Weather(scenario, seed=42)
        print(f"\n{scenario}:")
        print(f"  schedule: {w.schedule_summary()}")
        # Print condition at every 10th lap
        sample_laps = [1, 10, 20, 30, 40, 50, 52]
        for lap in sample_laps:
            print(f"    lap {lap:2d}: {w.condition_at_lap(lap)}")

    print("\n" + "-" * 60)
    print("Determinism check — same seed gives same schedule:")
    a = Weather("DryToWet", seed=123).schedule_summary()
    b = Weather("DryToWet", seed=123).schedule_summary()
    print(f"  Run A: {a}")
    print(f"  Run B: {b}")
    print(f"  Match: {a == b}")

    print("\nDifferent seeds give different schedules:")
    c = Weather("DryToWet", seed=123).schedule_summary()
    d = Weather("DryToWet", seed=456).schedule_summary()
    print(f"  seed=123: {c}")
    print(f"  seed=456: {d}")
    print(f"  Differ:  {c != d}")
