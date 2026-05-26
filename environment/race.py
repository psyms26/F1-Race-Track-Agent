# race.py
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import config
from environment.track import Track
from environment.weather import DRY, DRYING, WET, Weather
from environment.tyre import Tyre, compound_type
from environment.fuel import FuelTank


@dataclass
class Action:

    steering: float = 0.0
    throttle: float = 0.0
    pit: int = 0
    compound: str = "Medium"


class Race:

    def __init__(
        self,
        track: Track,
        weather: Weather,
        starting_compound: str,
        seed: Optional[int] = None,
    ):
        self.track = track
        self.weather = weather
        self.starting_compound = starting_compound
        self.seed = seed
        self.reset()

    def reset(self) -> "Race":
        # Car state
        self.s = 0.0
        self.lateral = 0.0
        self.speed = 0.0

        # Subsystems
        self.tyre = Tyre.new(self.starting_compound)
        self.fuel = FuelTank.new()

        # Progress
        self.lap = 1
        self.elapsed_time = 0.0
        self.lap_start_time = 0.0
        self.lap_times: List[float] = []
        self.current_condition = self.weather.update(self.lap)

        # Compliance tracking
        self.pit_stops = 0
        self.compounds_used: Set[str] = {self.starting_compound}
        self.used_wet_during_wet = (
            compound_type(self.starting_compound) == "Wets"
            and self.current_condition == WET
        )

        # Off-track
        self.offtrack_count = 0
        self.offtrack_penalty_added = 0.0
        self.was_off_track = False
        self.grip_recovery_remaining = 0.0

        # Pit request (latched until lap crossing)
        self.pit_requested = False
        self.next_pit_compound: Optional[str] = None

        # Termination
        self.dnf = False
        self.dnf_reason: Optional[str] = None
        self.finished = False
        return self

    #  main step

    def step(
        self, action: Action
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.dnf or self.finished:
            return self._observation(), 0.0, self.dnf, self.finished, self._info()

        dt = config.SIM_TIMESTEP
        reward = 0.0

        # 1. Latch pit request (resolved on next lap crossing)
        if action.pit == 1 and not self.pit_requested:
            self.pit_requested = True
            self.next_pit_compound = action.compound

        # 2. Compute grip from weather × compound × wear (with off-track recovery)
        grip = self._compute_grip()

        # 3. Determine max safe speed at current track position
        max_safe = self.track.max_safe_speed(self.s, grip)

        # 4. Apply throttle → adjust speed
        self._apply_throttle(action.throttle, max_safe, grip, dt)

        # 5. Apply steering & overshoot drift → update lateral
        self._apply_steering(action.steering, max_safe, dt)

        # 6. Advance position
        distance = self.speed * dt
        prev_s = self.s
        self.s = (self.s + distance) % self.track.length

        # 7. Off-track checks (entry, gravel, severe → DNF)
        in_gravel = self.track.is_in_gravel(self.lateral)
        is_off = self.track.is_off_track(self.lateral)
        self._handle_offtrack_transition(is_off, in_gravel, dt)
        if self.track.is_severe_offtrack(self.lateral):
            self._dnf("severe_offtrack")
            return self._observation(), config.REWARD_DNF, True, False, self._info()

        # 8. Tyre + fuel updates
        self.tyre.step(
            dt=dt,
            distance_covered=distance,
            speed=self.speed,
            max_safe_speed=max_safe,
            throttle=action.throttle,
            weather=self.current_condition,
            in_gravel=in_gravel,
        )
        self.fuel.step(distance_covered=distance, throttle=action.throttle)

        # 9. Time elapsed for this step (sim dt + fuel-weight time effect)
        step_time = dt + self.fuel.weight_time_effect(distance)
        self.elapsed_time += step_time
        if is_off:
            reward += config.REWARD_OFFTRACK_PER_SEC * dt

        # 10. Recovery countdown
        if self.grip_recovery_remaining > 0.0:
            self.grip_recovery_remaining = max(0.0, self.grip_recovery_remaining - dt)

        # 11. Out of fuel mid-race → forced DNF (can't move)
        if self.fuel.is_empty:
            self._dnf("out_of_fuel")
            return self._observation(), config.REWARD_DNF, True, False, self._info()

        # 12. Lap crossing (s wrapped from high to low)
        if self.s < prev_s:
            lap_reward = self._handle_lap_crossing()
            reward += lap_reward

        # 13. Continuous reward shaping

        reward += config.REWARD_WEAR_PER_PERCENT * 0.0  # tyre.step already updated wear

        return self._observation(), reward, self.dnf, self.finished, self._info()

    # physics helpers

    def _compute_grip(self) -> float:

        base = config.GRIP_MATRIX[self.current_condition][self.tyre.type]
        wear_factor = 1.0 - config.GRIP_WEAR_PENALTY * self.tyre.wear_fraction
        grip = base * wear_factor
        if self.grip_recovery_remaining > 0.0:
            grip *= config.OFFTRACK_GRIP_PENALTY
        return max(grip, config.GRIP_MIN)

    def _apply_throttle(
        self, throttle: float, max_safe: float, grip: float, dt: float
    ) -> None:

        speed_cap = min(config.MAX_SPEED * self.tyre.speed_factor, max_safe)
        if throttle >= 0.0:
            target = speed_cap * throttle
            if self.speed < target:
                self.speed = min(target, self.speed + config.ACCEL_RATE * dt)
            else:

                self.speed = max(target, self.speed - config.ACCEL_RATE * 0.5 * dt)
        else:

            self.speed = max(0.0, self.speed + config.BRAKE_RATE * throttle * dt)

    def _apply_steering(self, steering: float, max_safe: float, dt: float) -> None:

        speed_factor = min(self.speed / 20.0, 1.0)
        steer_velocity = steering * config.MAX_STEERING_ANGLE * 10.0 * speed_factor
        self.lateral += steer_velocity * dt


        curvature = self.track.curvature_at(self.s)
        if curvature != 0.0 and self.speed > max_safe:
            overshoot = (self.speed - max_safe) / max(max_safe, 1.0)
            outward = -1.0 if curvature > 0 else 1.0
            drift = outward * overshoot * 15.0 * dt
            self.lateral += drift

    def _handle_offtrack_transition(
        self, is_off: bool, in_gravel: bool, dt: float
    ) -> None:

        if is_off and not self.was_off_track:

            self.tyre.apply_offtrack_excursion()
            self.offtrack_count += 1
            if self.offtrack_count > config.OFFTRACK_FREE_EXCURSIONS:

                self.elapsed_time += config.OFFTRACK_TIME_PENALTY
                self.offtrack_penalty_added += config.OFFTRACK_TIME_PENALTY
        if not is_off and self.was_off_track:

            self.grip_recovery_remaining = config.OFFTRACK_GRIP_RECOVERY_TIME
        self.was_off_track = is_off

    # lap crossing & pit stop

    def _handle_lap_crossing(self) -> float:

        lap_time = self.elapsed_time - self.lap_start_time
        self.lap_times.append(lap_time)
        self.lap_start_time = self.elapsed_time
        reward = config.REWARD_LAP_BASE - lap_time


        if self.pit_requested:
            self._service_pit()


        # Advance lap and update weather for the new lap
        self.lap += 1
        if self.lap > config.TOTAL_LAPS:
            return reward + self._finish_race()

        prev_condition = self.current_condition
        self.current_condition = self.weather.update(self.lap)
        # Track if wet compound was used during any wet phase (for compliance)
        if (
            self.current_condition == WET
            and compound_type(self.tyre.compound) == "Wets"
        ):
            self.used_wet_during_wet = True

        return reward

    def _service_pit(self) -> None:

        pit_lane_time = config.PIT_LANE_LENGTH / max(config.PIT_LANE_SPEED_LIMIT, 1.0)
        self.elapsed_time += pit_lane_time + config.PIT_TYRE_CHANGE_TIME

        compound = self.next_pit_compound or self.tyre.compound
        self.tyre.reset(compound=compound)
        self.compounds_used.add(compound)
        self.pit_stops += 1
        self.pit_requested = False
        self.next_pit_compound = None

    # termination

    def _finish_race(self) -> float:

        bonus = config.REWARD_FINISH_BASE - self.elapsed_time
        # Mandatory minimum pit stops
        if self.pit_stops < config.MIN_PIT_STOPS:
            self._dnf("missed_mandatory_pit")
            return config.REWARD_DNF
        # Two-compound rule (dry races)
        if (
            config.DRY_REQUIRES_TWO_COMPOUNDS
            and not self.weather.has_any_wet_phase()
            and len(self.compounds_used) < 2
        ):
            self._dnf("missed_compound_diversity")
            return config.REWARD_DNF
        # Wet-compound rule (any race that had a wet phase)
        if (
            config.WET_REQUIRES_WET_COMPOUND
            and self.weather.has_any_wet_phase()
            and not self.used_wet_during_wet
        ):
            self._dnf("missed_wet_compound")
            return config.REWARD_DNF
        # Fuel sample rule
        if not self.fuel.passes_post_race_check():
            self._dnf("fuel_below_threshold")
            return config.REWARD_DNF

        self.finished = True
        return bonus

    def _dnf(self, reason: str) -> None:
        self.dnf = True
        self.dnf_reason = reason
        self.finished = False

    # observation & info

    def _observation(self) -> np.ndarray:

        speed_n = self.speed / max(config.MAX_SPEED, 1.0)
        wear_n = self.tyre.wear_fraction
        fuel_n = self.fuel.fraction_remaining
        laps_remaining = max(0, config.TOTAL_LAPS - self.lap + 1) / config.TOTAL_LAPS
        has_pitted = 1.0 if self.pit_stops > 0 else 0.0

        # Track condition one-hot (Dry, Drying, Wet)
        cond_oh = [0.0, 0.0, 0.0]
        cond_idx = {DRY: 0, DRYING: 1, WET: 2}[self.current_condition]
        cond_oh[cond_idx] = 1.0

        # Compound one-hot (S, M, H, W)
        comp_oh = [0.0, 0.0, 0.0, 0.0]
        comp_idx = {"Soft": 0, "Medium": 1, "Hard": 2, "Wet": 3}[self.tyre.compound]
        comp_oh[comp_idx] = 1.0

        # Curvature lookahead (5 points)
        lookahead = self.track.lookahead_curvatures(self.s, self.speed)

        # Distance to edges (normalised)
        d_left, d_right = self.track.distance_to_edges(self.lateral)
        half = self.track.width / 2.0
        d_left_n = max(-1.0, min(1.0, d_left / (half + self.track.gravel_width)))
        d_right_n = max(-1.0, min(1.0, d_right / (half + self.track.gravel_width)))

        obs = np.array(
            [speed_n, wear_n, fuel_n, laps_remaining, has_pitted]
            + cond_oh + comp_oh + list(lookahead) + [d_left_n, d_right_n],
            dtype=np.float32,
        )
        assert obs.shape == (config.RL_STATE_DIM,), (
            f"Observation shape mismatch: got {obs.shape}, expected "
            f"({config.RL_STATE_DIM},)"
        )
        return obs

    def _info(self) -> Dict[str, Any]:
        return {
            "lap": self.lap,
            "elapsed_time": self.elapsed_time,
            "speed": self.speed,
            "tyre_wear": self.tyre.wear,
            "tyre_compound": self.tyre.compound,
            "fuel": self.fuel.current,
            "weather": self.current_condition,
            "pit_stops": self.pit_stops,
            "compounds_used": sorted(self.compounds_used),
            "offtrack_count": self.offtrack_count,
            "lap_times": list(self.lap_times),
            "dnf": self.dnf,
            "dnf_reason": self.dnf_reason,
            "finished": self.finished,
        }

    # public helpers

    @property
    def grip(self) -> float:
        return self._compute_grip()

    @property
    def max_safe_speed(self) -> float:
        return self.track.max_safe_speed(self.s, self.grip)

    def __repr__(self) -> str:
        return (
            f"Race(lap={self.lap}/{config.TOTAL_LAPS}, "
            f"s={self.s:.0f}/{self.track.length:.0f}, "
            f"speed={self.speed:.1f}, "
            f"tyre={self.tyre.compound}@{self.tyre.wear:.0f}%, "
            f"fuel={self.fuel.current:.1f}, "
            f"weather={self.current_condition}, "
            f"time={self.elapsed_time:.1f}s)"
        )



# Sanity check


if __name__ == "__main__":
    print("Running a dummy full-throttle race (no steering, no pit logic)…\n")
    track = Track()
    weather = Weather("FullDry", seed=42)
    race = Race(track, weather, starting_compound="Medium")


    max_steps = int(config.TOTAL_LAPS * (track.length / 60.0) / config.SIM_TIMESTEP)
    step_count = 0

    while not race.dnf and not race.finished and step_count < max_steps:

        target_throttle = min(1.0, max(0.3, 0.85 * race.max_safe_speed / config.MAX_SPEED))

        target_lat = track.racing_line_lateral(race.s)
        steering = max(-1.0, min(1.0, (target_lat - race.lateral) * 0.3))

        pit = 1 if (race.lap == 30 and race.pit_stops == 0) else 0
        action = Action(steering=steering, throttle=target_throttle,
                        pit=pit, compound="Hard")
        race.step(action)
        step_count += 1
        if step_count % 5000 == 0:
            print(f"  step {step_count:6d}: {race}")

    print(f"\nFinal state: {race}")
    print(f"Result: {'DNF (' + str(race.dnf_reason) + ')' if race.dnf else 'FINISHED'}")
    print(f"Total race time: {race.elapsed_time:.2f} s")
    print(f"Laps completed: {len(race.lap_times)}")
    if race.lap_times:
        print(f"Mean lap time:  {sum(race.lap_times)/len(race.lap_times):.2f} s")
        print(f"Best lap:       {min(race.lap_times):.2f} s")
        print(f"Worst lap:      {max(race.lap_times):.2f} s")
    print(f"Pit stops: {race.pit_stops}, compounds used: "
          f"{sorted(race.compounds_used)}")
    print(f"Off-track count: {race.offtrack_count}, "
          f"penalty added: {race.offtrack_penalty_added:.1f} s")
    print(f"Fuel remaining: {race.fuel.current:.2f} ({race.fuel.fraction_remaining * 100:.1f}%)")
    print(f"Final tyre wear: {race.tyre.wear:.1f}%")