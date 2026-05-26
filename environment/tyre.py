# tyre.py
from dataclasses import dataclass
from typing import Optional
import config



def compound_type(compound: str) -> str:
    if compound in config.SLICKS:
        return "Slicks"
    if compound in config.WETS:
        return "Wets"

    raise ValueError(f"Unknown compound: {compound!r}")

def is_valid_compound(compound: str) -> bool:
    return compound in config.SLICKS or compound in config.WETS

def cornering_stress_factor(speed: float, max_safe_speed: float) -> float:
    if max_safe_speed <= 0:
        return config.TYRE_STRESS_MIN
    ratio = speed / max_safe_speed

    if ratio <= 0.7:
        return config.TYRE_STRESS_MIN
    if ratio >= 1.0:
        return config.TYRE_STRESS_MAX

    span = config.TYRE_STRESS_MAX - config.TYRE_STRESS_MIN
    return config.TYRE_STRESS_MIN + span * ((ratio - 0.7) / 0.3)

def braking_stress_factor(throttle: float) -> float:

    if throttle >= -0.5:
        return config.TYRE_STRESS_MIN
    if throttle <= -1.0:
        return config.TYRE_STRESS_MAX

    span = config.TYRE_STRESS_MAX - config.TYRE_STRESS_MIN
    return config.TYRE_STRESS_MIN + span * ((-throttle - 0.5) / 0.5)

def mismatch_factor(compound: str, weather: str) -> float:
    ctype = compound_type(compound)
    return config.TYRE_MISMATCH_FACTOR.get((ctype, weather), 1.0)


@dataclass
class Tyre:
    compound: str
    wear: float = 0.0
    age_steps: int = 0
    distance_covered: float = 0.0

    @classmethod
    def new(cls, compound: str) -> "Tyre":
        if not is_valid_compound(compound):
            raise ValueError(f"Invalid compound: {compound!r}")
        return cls(compound=compound)

    @property
    def type(self) -> str:
        return compound_type(self.compound)

    @property
    def wear_fraction(self) -> float:
        return self.wear / config.MAX_TYRE_WEAR

    @property
    def speed_factor(self) -> float:
        return config.TYRE_SPEED_FACTOR[self.compound]

    @property
    def is_worn_out(self) -> bool:
        return self.wear >= config.MAX_TYRE_WEAR

    @property
    def base_wear_per_lap(self) -> float:
        return config.TYRE_BASE_WEAR[self.compound]

    def step(
            self,
            dt: float,
            distance_covered: float,
            speed: float,
            max_safe_speed: float,
            throttle: float,
            weather: str,
            in_gravel: bool = False,
    ) -> float:

        if distance_covered <= 0:
            self.age_steps += 1
            return 0.0

        lap_fraction = distance_covered / config.TRACK_LENGTH
        base = self.base_wear_per_lap * lap_fraction

        stress = max(
            cornering_stress_factor(speed, max_safe_speed),
            braking_stress_factor(throttle),
        )

        mm = mismatch_factor(self.compound, weather)

        if in_gravel:
            offtrack_mult = 1.0 + config.OFFTRACK_WEAR_PER_SECOND * dt
        else:
            offtrack_mult = 1.0

        delta = base * stress * mm * offtrack_mult

        new_wear = min(self.wear + delta, config.MAX_TYRE_WEAR)
        actual_delta = new_wear - self.wear
        self.wear = new_wear
        self.age_steps += 1
        self.distance_covered += distance_covered
        return actual_delta

    def apply_offtrack_excursion(self) -> float:
        before = self.wear
        self.wear = min(
            self.wear + config.OFFTRACK_WEAR_PER_EXCURSION,
            config.MAX_TYRE_WEAR,
        )
        return self.wear - before

    def reset(self, compound: Optional[str] = None) -> None:
        if compound is not None:
            if not is_valid_compound(compound):
                raise ValueError(f"Invalid compound: {compound!r}")
            self.compound = compound
        self.wear = 0.0
        self.age_steps = 0
        self.distance_covered = 0.0

    def __repr__(self) -> str:
        return f"Tyre({self.compound}, wear={self.wear:.1f}%)"

# Sanity Check:
if __name__ == "__main__":
    print("Compound base wear rates per lap (no stress, no mismatch):")
    for c in ("Soft", "Medium", "Hard", "Wet"):
        print(f"  {c:6s}: {config.TYRE_BASE_WEAR[c]:.2f}% per lap")

    print("\nMismatch factors:")
    for (ctype, weather), f in config.TYRE_MISMATCH_FACTOR.items():
        print(f"  {ctype:6s} in {weather:6s}: ×{f}")

    print("\nStress factor curves:")
    print("  Cornering (speed / max_safe_speed):")
    for r in (0.5, 0.7, 0.8, 0.9, 1.0, 1.1):
        s = cornering_stress_factor(r * 60, 60.0)
        print(f"    {r:.2f}x safe → stress ×{s:.2f}")
    print("  Braking (throttle):")
    for thr in (0.0, -0.3, -0.5, -0.7, -0.9, -1.0):
        s = braking_stress_factor(thr)
        print(f"    throttle={thr:+.2f} → stress ×{s:.2f}")

    print("\nSimulating a Medium tyre across one full lap (no corners):")
    t = Tyre.new("Medium")
    dt = config.SIM_TIMESTEP
    speed = 80.0
    max_safe = 100.0
    distance_per_step = speed * dt
    steps_per_lap = int(config.TRACK_LENGTH / distance_per_step)
    total_delta = 0.0
    for _ in range(steps_per_lap):
        total_delta += t.step(
            dt=dt,
            distance_covered=distance_per_step,
            speed=speed,
            max_safe_speed=max_safe,
            throttle=0.5,
            weather="Dry",
            in_gravel=False,
        )
    print(f"  After 1 lap (cruising at 80% speed): wear = {t.wear:.2f}%")
    print(f"  Steps taken: {steps_per_lap}, total delta: {total_delta:.2f}%")

    print("\nWear projection — Medium in dry, normal driving:")
    t = Tyre.new("Medium")
    for lap in range(1, config.TOTAL_LAPS + 1):
        for _ in range(steps_per_lap):
            t.step(dt, distance_per_step, speed, max_safe, 0.5, "Dry", False)
        if lap in (1, 10, 20, 30, 40, 50, 52):
            print(f"  Lap {lap:2d}: wear = {t.wear:5.1f}%")

    print("\nMismatch test — Slicks driven in Wet weather:")
    t = Tyre.new("Medium")
    for lap in range(1, 11):
        for _ in range(steps_per_lap):
            t.step(dt, distance_per_step, speed, max_safe, 0.5, "Wet", False)
        if lap in (1, 5, 10):
            print(f"  Lap {lap:2d}: wear = {t.wear:5.1f}% (mismatch ×3.0 active)")

    print("\nOff-track excursion test:")
    t = Tyre.new("Soft")
    t.wear = 30.0
    print(f"  Before excursion: {t.wear:.1f}%")
    delta = t.apply_offtrack_excursion()
    print(f"  After excursion:  {t.wear:.1f}% (added {delta:.1f}%)")

    print("\nPit stop reset test:")
    t = Tyre.new("Soft")
    t.wear = 75.0
    print(f"  Before pit: {t}")
    t.reset(compound="Hard")
    print(f"  After pit:  {t}")
