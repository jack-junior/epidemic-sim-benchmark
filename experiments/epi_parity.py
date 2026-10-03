"""Epidemiological parity report: ECS (Esper) vs Mesa over independent replicates.

    uv run python experiments/epi_parity.py --replicates 100 --n-agents 300 --scenario hazard_similarity

Writes results/epi_parity_<scenario>.json and prints a table.
"""
import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np

from epidemic_sim.analysis.parity import ESTIMANDS, compare_estimand, summarise
from epidemic_sim.ecs.simulation import SIREpidemicModel
from epidemic_sim.mesa import MesaSIREpidemicModel

SCENARIOS = {
    "hazard_similarity": dict(spatial_new=True, network_new=True,
                              space_attribute_similarity=True, dispersion=0.65),
    "hazard_random": dict(spatial_new=True, network_new=True, space_attribute_similarity=False),
    "sequential_random": dict(spatial_new=False, network_new=False, space_attribute_similarity=False),
    "hazard_quarantine": dict(spatial_new=True, network_new=True, enable_quarantine=True,
                              space_attribute_similarity=True, dispersion=0.65),
}


def replicate(scenario: dict, replicates: int, n_agents: int, world_size: int, max_steps: int,
              first_seed: int = 1000) -> dict[str, list[dict[str, float]]]:
    base = dict(n_agents=n_agents, world_size=world_size, initial_infected=max(3, n_agents // 40),
                beta_spatial=0.3, beta_network=0.25)
    out: dict[str, list[dict[str, float]]] = {"ecs": [], "mesa": []}
    for r in range(replicates):
        with contextlib.redirect_stdout(io.StringIO()):
            ecs = SIREpidemicModel(seed=first_seed + r, **base, **scenario)
            ecs.run(max_steps)
        out["ecs"].append(summarise(ecs.time_series_data, n_agents))
        ecs.clean_up()
        # independent seeds for the two implementations: replicates are independent samples
        model = MesaSIREpidemicModel(seed=first_seed + 100_000 + r, collect_spatial=False,
                                     **base, **scenario)
        with contextlib.redirect_stdout(io.StringIO()):
            model.run(max_steps)
        out["mesa"].append(summarise(model.time_series_data, n_agents))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="hazard_similarity")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=300)
    parser.add_argument("--world-size", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--first-seed", type=int, default=1000)
    args = parser.parse_args()

    start = time.perf_counter()
    runs = replicate(SCENARIOS[args.scenario], args.replicates, args.n_agents,
                     args.world_size, args.max_steps, args.first_seed)
    rng = np.random.default_rng(0)
    report = {m: compare_estimand([r[m] for r in runs["ecs"]], [r[m] for r in runs["mesa"]], rng)
              for m in ESTIMANDS}
    print(f"{args.scenario}: {args.replicates} replicates, N={args.n_agents}, "
          f"{time.perf_counter() - start:.0f}s")
    print(f"{'estimand':16s} {'ECS mean (sd)':>16s} {'Mesa mean (sd)':>16s} "
          f"{'Welch p':>8s} {'KS p':>7s} {'BF p':>7s} {'var ratio [95% CI]':>24s}  parity")
    for m, r in report.items():
        print(f"{m:16s} {r['mean_ecs']:8.3f} ({r['sd_ecs']:.3f}) {r['mean_mesa']:8.3f} ({r['sd_mesa']:.3f}) "
              f"{r['p_mean_welch']:8.3f} {r['p_ks']:7.3f} {r['p_variance_brown_forsythe']:7.3f} "
              f"{r['variance_ratio']:8.2f} [{r['variance_ratio_low']:.2f}, {r['variance_ratio_high']:.2f}]"
              f"  {'OK' if r['parity'] else 'DIFFERENT'}")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/epi_parity_{args.scenario}.json").write_text(
        json.dumps({"args": vars(args), "report": report, "runs": runs}, default=float, indent=1))


if __name__ == "__main__":
    main()
