"""Exactness of the vectorised kernels against the plain definitions."""
from itertools import permutations

import numpy as np
import pytest

from epidemic_sim.optim import kernels


def proximity_loops(s_pos, s_imm, i_pos, i_load, radius, beta):
    out = np.zeros(len(s_pos))
    for s in range(len(s_pos)):
        for i in range(len(i_pos)):
            if np.hypot(*(i_pos[i] - s_pos[s])) <= radius:
                out[s] += (beta * i_load[i] / 1000) * (1 - s_imm[s])
    return out


@pytest.mark.parametrize("world", [60, 200])
def test_proximity_hazard_equals_the_double_loop(world):
    rng = np.random.default_rng(0)
    s_pos, i_pos = rng.uniform(0, world, (300, 2)), rng.uniform(0, world, (40, 2))
    s_imm, i_load = rng.uniform(0, 0.05, 300), rng.uniform(500, 1000, 40)
    fast = kernels.proximity_hazard(s_pos, s_imm, i_pos, i_load, 4.0, 0.3)
    assert np.allclose(fast, proximity_loops(s_pos, s_imm, i_pos, i_load, 4.0, 0.3), atol=1e-12)


def test_proximity_hazard_without_infectious_or_susceptible_is_zero():
    pos = np.zeros((3, 2))
    assert not kernels.proximity_hazard(pos, np.zeros(3), np.zeros((0, 2)), np.zeros(0), 4, 0.1).any()
    assert kernels.proximity_hazard(np.zeros((0, 2)), np.zeros(0), pos, np.ones(3), 4, 0.1).size == 0


def exact_inclusion(weights: np.ndarray, k: int) -> np.ndarray:
    """P(j in the successive weighted sample of size k), by enumeration of ordered k-tuples."""
    total, n = weights.sum(), len(weights)
    inclusion = np.zeros(n)
    for ordered in permutations(range(n), k):
        probability, remaining = 1.0, total
        for j in ordered:
            probability *= weights[j] / remaining
            remaining -= weights[j]
        for j in ordered:
            inclusion[j] += probability
    return inclusion


@pytest.mark.parametrize("sampler", [kernels.rewire_rejection, kernels.rewire_gumbel])
def test_rewiring_is_the_successive_weighted_sample(sampler):
    rng = np.random.default_rng(1)
    n, k, draws = 7, 3, 3000
    x, y = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    age = rng.integers(0, 75, n).astype(float)
    alive = np.ones(n, dtype=bool)
    alpha, tau = 40.0, 25.0
    counts = np.zeros((n, n))
    for _ in range(draws):
        src, dst, _w = sampler(x, y, age, alive, np.full(n, k), alpha, tau, rng)
        assert len(src) == n * k and len(set(zip(src, dst))) == n * k and (src != dst).all()
        np.add.at(counts, (src, dst), 1)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        w = np.exp(-np.hypot(x[i] - x[others], y[i] - y[others]) / alpha
                   - np.abs(age[i] - age[others]) / tau)
        expected = exact_inclusion(w, k)
        observed = counts[i, others] / draws
        sigma = np.sqrt(expected * (1 - expected) / draws)
        assert np.all(np.abs(observed - expected) < 5 * sigma + 1e-3)


def test_rewiring_excludes_dead_agents_and_returns_the_similarity_as_strength():
    rng = np.random.default_rng(2)
    n = 60
    x, y = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    age = rng.integers(0, 75, n).astype(float)
    alive = np.arange(n) % 5 != 0
    src, dst, w = kernels.rewire_rejection(x, y, age, alive, np.full(n, 4), 20.0, 25.0, rng)
    assert alive[src].all() and alive[dst].all()
    expected = np.exp(-np.hypot(x[src] - x[dst], y[src] - y[dst]) / 20.0
                      - np.abs(age[src] - age[dst]) / 25.0)
    assert np.allclose(w, expected)


def test_rewiring_caps_the_number_of_contacts_at_the_population():
    rng = np.random.default_rng(3)
    x, y, age = rng.uniform(0, 10, 5), rng.uniform(0, 10, 5), np.arange(5.0)
    src, dst, _ = kernels.rewire_rejection(x, y, age, np.ones(5, bool), np.full(5, 50), 5.0, 25.0, rng)
    assert len(src) == 5 * 4
