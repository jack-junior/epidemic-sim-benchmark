"""Temps passé dans chaque système, ECS (Esper) vs Mesa, sur un nombre de pas FIXE.

    uv run python experiments/profile_models.py                 # 1000 agents, 50 pas
    uv run python experiments/profile_models.py --n-agents 2000 --steps 40

Même configuration que la démo de scripts/run_ecs.py. Les diagnostics d'affichage de l'ECS
sont coupés pour ne pas fausser la mesure. Résultat = une seule exécution : à répéter
plusieurs fois avant d'en tirer des conclusions.
"""
import argparse
import contextlib
import io
import time
from collections import defaultdict

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
        "MovementSystem": "1 déplacement",
        "NetworkRewiringystem": "2 re-câblage du réseau",
        "SpatialTransmissionSystemNew": "3 contamination par proximité",
        "NetworkTransmissionSystemNew": "4 contamination par les contacts",
        "InfectionResolutionSystem": "5 tirage des nouvelles infections",
        "DiseaseProgressionSystem": "6 évolution de la maladie",
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
    Person.move = timed(Person.move, "1 déplacement", totals)
    Person.progress_disease = timed(Person.progress_disease, "6 évolution de la maladie", totals)
    for name, label in {
        "_rewire_network": "2 re-câblage du réseau",
        "_spatial_transmission_hazard": "3 contamination par proximité",
        "_network_transmission_hazard": "4 contamination par les contacts",
        "_resolve_infections": "5 tirage des nouvelles infections",
    }.items():
        setattr(model, name, timed(getattr(model, name), label, totals))
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            model.step()
    return time.perf_counter() - start, totals


def show(title, total, totals):
    print(f"\n{title}  (total {total:.2f} s)")
    for label in sorted(totals):
        print(f"  {label:38s} {totals[label]:7.2f} s  {100 * totals[label] / total:5.1f} %")
    rest = total - sum(totals.values())
    print(f"  {'reste (collecte des données, etc.)':38s} {rest:7.2f} s  {100 * rest / total:5.1f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    print(f"{args.n_agents} agents, {args.steps} pas (nombre de pas fixe)")
    show("ECS (Esper)", *profile_ecs(args.n_agents, args.steps))
    show("Mesa (objets)", *profile_mesa(args.n_agents, args.steps))
