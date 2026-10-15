"""Time spent in each system, ECS (Esper) vs Mesa, for a fixed number of steps.

    uv run python -m experiments.profiling.profile_models
    uv run python -m experiments.profiling.profile_models --n-agents 2000 --steps 40

Uses the same configuration as ``scripts/run_ecs.py``. ECS diagnostic output is
disabled to avoid skewing the measurements. This is a single run; repeat it
before drawing conclusions.
"""
import argparse
import contextlib
import io
import sys
import time
from collections import defaultdict
from collections.abc import Sequence

import esper

from epidemic_sim.ecs.simulation import SIREpidemicModel
from epidemic_sim.ecs.systems import transmission as ecs_systems
from epidemic_sim.mesa import MesaSIREpidemicModel
from epidemic_sim.mesa.agents import Person


def config(n_agents):
    return dict(seed=28022026, enable_quarantine=False, initial_infected=20,
                world_size=500, beta_spatial=0.3, beta_network=0.25,
                spatial_new=True, network_new=True, space_attribute_similarity=True,
                n_agents=n_agents, dt=1.0, dispersion=0.65)


def timed(function, label, totals):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            totals[label] += time.perf_counter() - start
    return wrapper


def profile_ecs(n_agents, steps):
    with contextlib.redirect_stdout(io.StringIO()):
        model = SIREpidemicModel(**config(n_agents))
    esper.switch_world(model.world_name)
    totals = defaultdict(float)
    labels = {
        "MovementSystem": "1 movement",
        "NetworkRewiringystem": "2 network rewiring",
        "SpatialTransmissionSystemNew": "3 spatial transmission",
        "NetworkTransmissionSystemNew": "4 network transmission",
        "InfectionResolutionSystem": "5 infection resolution",
        "DiseaseProgressionSystem": "6 disease progression",
    }
    for class_name, label in labels.items():
        processor = esper.get_processor(getattr(ecs_systems, class_name))
        if class_name == "NetworkRewiringystem":
            processor.diagnostic = False
        processor.process = timed(processor.process, label, totals)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            model.step()
    return time.perf_counter() - start, totals


def profile_mesa(n_agents, steps):
    model = MesaSIREpidemicModel(**config(n_agents))
    totals = defaultdict(float)
    Person.progress_disease = timed(Person.progress_disease, "6 disease progression", totals)
    for name, label in {
        "_move": "1 movement",
        "_rewire_network": "2 network rewiring",
        "_spatial_transmission_hazard": "3 spatial transmission",
        "_network_transmission_hazard": "4 network transmission",
        "_resolve_infections": "5 infection resolution",
    }.items():
        setattr(model, name, timed(getattr(model, name), label, totals))
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            model.run_for(1)
    return time.perf_counter() - start, totals


def show(title, total, totals):
    print(f"\n{title}  (total {total:.2f} s)")
    for label in sorted(totals):
        print(f"  {label:38s} {totals[label]:7.2f} s  {100 * totals[label] / total:5.1f} %")
    rest = total - sum(totals.values())
    print(f"  {'other (data collection, etc.)':38s} {rest:7.2f} s  {100 * rest / total:5.1f} %")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-agents", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args(argv)
    print(f"{args.n_agents} agents, {args.steps} steps (fixed count)")
    show("ECS (Esper)", *profile_ecs(args.n_agents, args.steps))
    show("Mesa (objets)", *profile_mesa(args.n_agents, args.steps))


if __name__ == "__main__":
    main(sys.argv[1:])
