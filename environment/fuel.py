# Fuel.py
from dataclasses import dataclass
import config

@dataclass
class FuelTank:

    capacity: float = config.FUEL_START
    current: float = config.FUEL_START

    @classmethod
    def new(cls) -> "FuelTank":
        return cls(capacity=config.FUEL_START, current=config.FUEL_START)

    @property
    def fraction_remaining(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return max(0.0, self.current / self.capacity)

    @property
    def is_empty(self) -> bool:
        return self.current == 0.0

    @property
    def below_dnf_threshold(self) -> bool:
        return self.current < config.FUEL_DNF_THRESHOLD

    def passes_post_race_check(self) -> bool:
        return self.current >= config.FUEL_DNF_THRESHOLD

    def step(
            self,
            distance_covered: float,
            throttle: float,
    ) -> float:
        """Consume fuel per metre travelled (scale-invariant across tracks)."""
        if distance_covered <= 0:
            return 0.0

        base = config.FUEL_BASE_CONSUMPTION_PER_METER * distance_covered
        throttle_mult = 1.0 + config.FUEL_THROTTLE_FACTOR * max(0.0, throttle)
        delta = base * throttle_mult

        new_level = max(0.0, self.current - delta)
        actual_delta = self.current - new_level
        self.current = new_level
        return actual_delta

    def weight_time_effect(self, distance_covered: float) -> float:
        """Per-metre weight-related lap-time effect (scale-invariant)."""
        if distance_covered <= 0:
            return 0.0
        return config.FUEL_WEIGHT_TIME_PER_METER * self.current * distance_covered

    def __repr__(self) -> str:
        return (
            f"FuelTank({self.current:.1f}/{self.capacity:.1f} units, "
            f"{self.fraction_remaining * 100:.1f}%)"
        )


# Sanity Check
if __name__ == "__main__":
    print("Fuel system parameters:")
    print(f"  Starting fuel:        {config.FUEL_START}")
    print(f"  DNF threshold:        {config.FUEL_DNF_THRESHOLD} "
          f"({config.FUEL_DNF_THRESHOLD / config.FUEL_START * 100:.0f}%)")
    print(f"  Base consumption:     {config.FUEL_BASE_CONSUMPTION_PER_METER} per metre")
    print(f"  Throttle factor:      +{config.FUEL_THROTTLE_FACTOR}× per positive throttle (fuel cut on brake)")
    print(f"  Weight time cost:     {config.FUEL_WEIGHT_TIME_PER_METER} sec/(metre × unit)")

    base_per_lap = config.FUEL_BASE_CONSUMPTION_PER_METER * config.TRACK_LENGTH
    print(f"\nEquivalent per-lap consumption on {config.TRACK_LENGTH}m track: {base_per_lap:.3f}/lap")

    print("\nConsumption per lap at different throttle levels:")
    for thr in (0.0, 0.3, 0.5, 0.7, 1.0):
        per_lap = base_per_lap * (1.0 + config.FUEL_THROTTLE_FACTOR * max(0.0, thr))
        full_race = per_lap * config.TOTAL_LAPS
        print(
            f"  throttle={thr:.1f}: {per_lap:.3f}/lap  →  "
            f"{full_race:.1f} total over {config.TOTAL_LAPS} laps  "
            f"(margin: {config.FUEL_START - full_race:+.1f})"
        )

    print("\nFull-race simulation at moderate throttle (0.5):")
    tank = FuelTank.new()
    dt = config.SIM_TIMESTEP
    speed = 80.0
    distance_per_step = speed * dt
    steps_per_lap = int(config.TRACK_LENGTH / distance_per_step)
    cumulative_weight_effect = 0.0
    for lap in range(1, config.TOTAL_LAPS + 1):
        for _ in range(steps_per_lap):
            tank.step(distance_per_step, throttle=0.5)
            cumulative_weight_effect += tank.weight_time_effect(distance_per_step)
        if lap in (1, 10, 20, 30, 40, 50, 52):
            print(
                f"  Lap {lap:2d}: {tank.current:6.2f} units "
                f"({tank.fraction_remaining * 100:5.1f}%) "
                f"cumulative weight effect: {cumulative_weight_effect:.1f}s"
            )

    print(f"\nPost-race check: passes = {tank.passes_post_race_check()} "
          f"(needs ≥ {config.FUEL_DNF_THRESHOLD})")

    print("\nAggressive driving simulation (throttle 1.0):")
    tank2 = FuelTank.new()
    for lap in range(1, config.TOTAL_LAPS + 1):
        for _ in range(steps_per_lap):
            tank2.step(distance_per_step, throttle=1.0)
        if lap in (1, 30, 50, 52) or tank2.is_empty:
            print(
                f"  Lap {lap:2d}: {tank2.current:6.2f} units "
                f"({tank2.fraction_remaining * 100:5.1f}%)"
            )
            if tank2.is_empty:
                print("    → ran dry mid-race!")
                break
    print(f"  Post-race check: passes = {tank2.passes_post_race_check()}")

    print("\nWeight effect demonstration (lap-time impact on current track):")
    weight_per_lap_per_unit = config.FUEL_WEIGHT_TIME_PER_METER * config.TRACK_LENGTH
    for fuel_level in (110.0, 80.0, 50.0, 20.0, 11.0):
        per_lap = weight_per_lap_per_unit * fuel_level
        print(f"  Fuel={fuel_level:6.1f}: +{per_lap:.2f} sec/lap")
    diff = weight_per_lap_per_unit * 110.0 - weight_per_lap_per_unit * 11.0
    print(f"  Full → near-empty difference: {diff:.2f} sec/lap")
