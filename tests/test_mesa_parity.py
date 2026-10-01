"""Parity tests between the ECS (Esper) and the object-oriented (Mesa) models.

Two levels of parity are checked:

* exact parity of the initialisation (same seed -> same population, same initial
  infections, same contact network) and of the first movement step;
* statistical parity of the epidemic dynamics over replicated runs.  Exact
  trajectories cannot be identical because Esper iterates over entities in
  hash-set order, which changes which agent receives which random draw.
"""
import contextlib
import io
import subprocess
import sys

import esper
import numpy as np
import pytest
from scipy.stats import ks_2samp

from epidemic_sim.ecs.components import (
    ContactNetwork, Demographics, Infected, Location, Susceptible,
)
from epidemic_sim.ecs.simulation import SIREpidemicModel
from epidemic_sim.mesa import MesaSIREpidemicModel


def build_ecs(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return SIREpidemicModel(**kwargs)


def ecs_state(model):
    esper.switch_world(model.world_name)
    state = {}
    for entity in model.entity_iDs:
        location = esper.component_for_entity(entity, Location)
        demographics = esper.component_for_entity(entity, Demographics)
        if esper.has_component(entity, Susceptible):
            health = ("S", esper.component_for_entity(entity, Susceptible).immunity)
        else:
            infected = esper.component_for_entity(entity, Infected)
            health = ("I", infected.viral_load, infected.recovery_time, infected.days_infected)
        network = (esper.component_for_entity(entity, ContactNetwork)
                   if esper.has_component(entity, ContactNetwork) else None)
        contacts = [int(c) for c in network.contacts] if network else []
        strengths = list(network.contact_strength) if network else []
        state[entity] = (location.x, location.y, int(demographics.age),
                         demographics.mobility, health, contacts, strengths)
    return state


def mesa_state(model):
    state = {}
    for person in model.agents:
        if person.state == "S":
            health = ("S", person.immunity)
        else:
            health = ("I", person.viral_load, person.recovery_time, person.days_infected)
        state[person.unique_id] = (person.x, person.y, person.age, person.mobility, health,
                                   [c.unique_id for c in person.contacts],
                                   list(person.contact_strength))
    return state


@pytest.mark.parametrize("similarity", [True, False])
def test_initial_state_is_identical(similarity):
    kwargs = dict(seed=28022026, n_agents=200, world_size=200, initial_infected=8,
                  spatial_new=True, network_new=True,
                  space_attribute_similarity=similarity, dispersion=0.65)
    ecs = build_ecs(**kwargs)
    mesa_model = MesaSIREpidemicModel(**kwargs)
    assert ecs_state(ecs) == mesa_state(mesa_model)


def test_first_movement_is_identical():
    kwargs = dict(seed=7, n_agents=200, world_size=200, initial_infected=5,
                  spatial_new=True, network_new=True)
    ecs = build_ecs(**kwargs)
    mesa_model = MesaSIREpidemicModel(**kwargs)
    esper.switch_world(ecs.world_name)
    ecs.step()
    mesa_model.step()
    a, b = ecs_state(ecs), mesa_state(mesa_model)
    assert all(a[k][:2] == b[k][:2] for k in a)


def test_mesa_model_is_reproducible():
    kwargs = dict(seed=11, n_agents=150, world_size=150, initial_infected=5,
                  spatial_new=True, network_new=True, space_attribute_similarity=True,
                  dispersion=0.65)
    runs = []
    for _ in range(2):
        model = MesaSIREpidemicModel(**kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            model.run(40)
        runs.append(dict(model.time_series_data))
    assert runs[0] == runs[1]


def test_different_seeds_give_different_runs():
    kwargs = dict(n_agents=150, world_size=150, initial_infected=5, spatial_new=True, network_new=True)
    a = MesaSIREpidemicModel(seed=1, **kwargs)
    b = MesaSIREpidemicModel(seed=2, **kwargs)
    assert mesa_state(a) != mesa_state(b)


def test_population_is_conserved():
    model = MesaSIREpidemicModel(seed=3, n_agents=200, world_size=150, initial_infected=6,
                                 spatial_new=True, network_new=True, enable_quarantine=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model.run(60)
    series = model.time_series_data
    for i in range(len(series["time"])):
        total = (series["susceptible"][i] + series["infected"][i]
                 + series["recovered"][i] + series["death"][i])
        assert total == 200


def test_quarantine_reduces_mobility_and_restores_it():
    model = MesaSIREpidemicModel(seed=5, n_agents=20, world_size=50, initial_infected=1,
                                 spatial_new=True, network_new=True)
    person = next(a for a in model.agents if a.state == "I")
    original = person.mobility
    person.try_quarantine(compliance=1.0)  # certain to enter quarantine
    assert person.quarantined
    assert person.mobility == pytest.approx(original * (1 - 0.9))
    for _ in range(14):
        person.tick_quarantine()
    assert not person.quarantined
    assert person.mobility == pytest.approx(original)


@pytest.mark.parametrize("bad", [
    dict(n_agents=0), dict(initial_infected=999), dict(world_size=0), dict(dt=0),
    dict(beta_spatial=1.5), dict(tau=0), dict(dispersion=0), dict(alpha=-1),
])
def test_invalid_parameters_are_rejected(bad):
    with pytest.raises(ValueError):
        MesaSIREpidemicModel(seed=1, **bad)


def test_seed_must_be_integer():
    with pytest.raises(TypeError):
        MesaSIREpidemicModel(seed=1.5)


def test_mesa_module_does_not_import_esper():
    """A Mesa-only run must not load Esper (fair memory measurement in the benchmark)."""
    code = "import sys, epidemic_sim.mesa; assert 'esper' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


# ----------------------------------------------------------------------------
# Statistical parity over replicated runs
# ----------------------------------------------------------------------------
SCENARIOS = {
    "hazard_random": dict(spatial_new=True, network_new=True, space_attribute_similarity=False),
    "sequential_random": dict(spatial_new=False, network_new=False, space_attribute_similarity=False),
    "hazard_quarantine": dict(spatial_new=True, network_new=True, enable_quarantine=True,
                              space_attribute_similarity=True, dispersion=0.65),
}


def summarise(series, n_agents):
    infected = series["infected"]
    return dict(
        attack=(series["recovered"][-1] + series["death"][-1] + infected[-1]) / n_agents,
        peak=max(infected),
        t_peak=infected.index(max(infected)),
        length=len(infected),
    )


def replicate(scenario, replicates, n_agents=150, world_size=150):
    base = dict(n_agents=n_agents, world_size=world_size, initial_infected=6,
                beta_spatial=0.3, beta_network=0.25)
    results = {"ecs": [], "mesa": []}
    for r in range(replicates):
        seed = 500 + r
        ecs = build_ecs(seed=seed, **base, **scenario)
        with contextlib.redirect_stdout(io.StringIO()):
            ecs.run(300)
        results["ecs"].append(summarise(ecs.time_series_data, n_agents))
        ecs.clean_up()
        model = MesaSIREpidemicModel(seed=seed, collect_spatial=False, **base, **scenario)
        with contextlib.redirect_stdout(io.StringIO()):
            model.run(300)
        results["mesa"].append(summarise(model.time_series_data, n_agents))
    return results


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_epidemic_dynamics_are_statistically_equivalent(name):
    results = replicate(SCENARIOS[name], replicates=30)
    for metric in ("attack", "peak", "t_peak", "length"):
        ecs = np.array([r[metric] for r in results["ecs"]], dtype=float)
        mesa_values = np.array([r[metric] for r in results["mesa"]], dtype=float)
        p_value = ks_2samp(ecs, mesa_values).pvalue
        assert p_value > 0.01, (f"{name}/{metric}: ECS mean {ecs.mean():.2f} vs "
                                f"Mesa mean {mesa_values.mean():.2f} (KS p={p_value:.4f})")
