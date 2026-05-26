import argparse
import csv
import os
import time
import traceback
from datetime import datetime
import config
from environment.track import Track
from environment.weather import Weather
from environment.race import Race
from agents.rule_based import RuleBasedAgent
from agents.utility import UtilityAgent
from agents.rl_agent import RLAgent


# Map agent name → class
AGENT_CLASSES = {
    "RuleBased": RuleBasedAgent,
    "Utility":   UtilityAgent,
    "RL":        RLAgent,
}

CSV_FIELDS = [
    "agent", "scenario", "seed",
    "finished", "dnf", "dnf_reason",
    "total_time_s", "total_time_min",
    "laps_completed",
    "mean_lap_time_s", "best_lap_s", "worst_lap_s", "lap_time_std_s",
    "pit_stops", "compounds_used", "n_compounds",
    "off_track_count",
    "fuel_remaining_kg", "final_tyre_wear_pct",
]


def _variance(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _std(values):
    return _variance(values) ** 0.5


def run_single_race(agent_class, scenario, seed, max_steps=200_000):

    track = Track()
    weather = Weather(scenario, seed=seed)
    agent = agent_class(weather_forecast=weather.forecast_starting_condition())
    race = Race(track, weather, starting_compound=agent.starting_tyre)

    step_count = 0
    while not race.dnf and not race.finished and step_count < max_steps:
        action = agent.select_action(race)
        race.step(action)
        step_count += 1

    lap_times = race.lap_times
    mean_lap = sum(lap_times) / len(lap_times) if lap_times else 0.0
    best_lap = min(lap_times) if lap_times else 0.0
    worst_lap = max(lap_times) if lap_times else 0.0
    lap_std = _std(lap_times) if lap_times else 0.0

    return {
        "agent":              agent.name,
        "scenario":           scenario,
        "seed":               seed,
        "finished":           race.finished,
        "dnf":                race.dnf,
        "dnf_reason":         race.dnf_reason or "",
        "total_time_s":       round(race.elapsed_time, 3),
        "total_time_min":     round(race.elapsed_time / 60.0, 3),
        "laps_completed":     len(lap_times),
        "mean_lap_time_s":    round(mean_lap, 3),
        "best_lap_s":         round(best_lap, 3),
        "worst_lap_s":        round(worst_lap, 3),
        "lap_time_std_s":     round(lap_std, 3),
        "pit_stops":          race.pit_stops,
        "compounds_used":     "+".join(sorted(race.compounds_used)),
        "n_compounds":        len(race.compounds_used),
        "off_track_count":    race.offtrack_count,
        "fuel_remaining_kg":  round(race.fuel.current, 3),
        "final_tyre_wear_pct": round(race.tyre.wear, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int,
                        default=config.EXPERIMENT_RUNS_PER_CONDITION,
                        help="Seeds per (scenario, agent) — default 15")
    parser.add_argument("--scenarios", nargs="+",
                        default=config.EXPERIMENT_CONDITIONS,
                        choices=config.WEATHER_SCENARIOS)
    parser.add_argument("--agents", nargs="+",
                        default=config.EXPERIMENT_AGENTS,
                        choices=list(AGENT_CLASSES.keys()))
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: results/experiments_<ts>.csv)")
    parser.add_argument("--max-steps", type=int, default=200_000)
    args = parser.parse_args()

    # Output path
    if args.output is None:
        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/experiments_{ts}.csv"


    seeds = [config.BASE_SEED + i for i in range(args.runs)]

    total = len(args.scenarios) * len(args.agents) * len(seeds)
    print(f"\n{'='*80}")
    print(f" DIA F1 EXPERIMENT RUNNER")
    print(f"{'='*80}")
    print(f" Scenarios:  {args.scenarios}")
    print(f" Agents:     {args.agents}")
    print(f" Seeds:      {seeds}")
    print(f" Total runs: {total}")
    print(f" Output:     {args.output}")
    print(f"{'='*80}\n")

    start = time.time()
    failures = 0

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        i = 0
        for scenario in args.scenarios:
            for agent_name in args.agents:
                agent_class = AGENT_CLASSES[agent_name]
                for seed in seeds:
                    i += 1
                    t0 = time.time()
                    try:
                        result = run_single_race(
                            agent_class, scenario, seed, max_steps=args.max_steps
                        )
                        writer.writerow(result)
                        f.flush()

                        status = "FIN" if result["finished"] else f"DNF[{result['dnf_reason']}]"
                        dt = time.time() - t0
                        print(f"[{i:3d}/{total}] {agent_name:10s} {scenario:10s} "
                              f"seed={seed:4d}  {status:25s} "
                              f"time={result['total_time_min']:5.1f}min  "
                              f"laps={result['laps_completed']:2d}  "
                              f"pits={result['pit_stops']}  "
                              f"offT={result['off_track_count']:2d}  "
                              f"({dt:4.1f}s)")
                    except Exception as e:
                        failures += 1
                        print(f"[{i:3d}/{total}] {agent_name} {scenario} seed={seed}  "
                              f"ERROR: {e}")
                        traceback.print_exc()

    elapsed = time.time() - start
    print(f"\n{'='*80}")
    print(f" DONE — {total} races completed in {elapsed/60:.1f} min "
          f"({elapsed/total:.1f}s/race avg)")
    if failures:
        print(f" Failures: {failures}")
    print(f" Results saved to: {args.output}")
    print(f"{'='*80}\n")
    print(f"Next step:  python -m analysis.analyse_results {args.output}")


if __name__ == "__main__":
    main()