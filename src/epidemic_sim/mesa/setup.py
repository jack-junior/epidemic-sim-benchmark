"""Fixed definitions of the Mesa implementation: health states, model constants,
default parameters and the initial health profile of the population.

Everything that does not change during a run lives here instead of being hard-coded
inside the agent and model classes.
"""

from enum import Enum
from typing import Mapping, Optional


class State(Enum):
    """Health state of a person."""

    Susceptible = "S"
    Infected = "I"
    Recovered = "R"
    Dead = "D"


# Names used in the output series (same keys as the ECS model).
STATE_SERIES_NAMES: dict[State, str] = {
    State.Susceptible: "susceptible",
    State.Infected: "infected",
    State.Recovered: "recovered",
    State.Dead: "death",
}

# Population drawn at initialisation (same distributions as the ECS model).
AGE_RANGE: tuple[int, int] = (0, 75)           # integer ages in [0, 75)
MOBILITY_RANGE: tuple[float, float] = (0.5, 20.0)
IMMUNITY_RANGE: tuple[float, float] = (0.0, 0.05)
IMMUNITY_AFTER_RECOVERY: float = 0.9

# Disease parameters (same as the ECS systems).
INITIAL_VIRAL_LOAD_RANGE: tuple[float, float] = (700.0, 1000.0)
NEW_VIRAL_LOAD_RANGE: tuple[float, float] = (500.0, 1000.0)
RECOVERY_TIME_MEAN: float = 12.0
RECOVERY_TIME_SD: float = 4.0
RECOVERY_PROBABILITY_RANGE: tuple[float, float] = (0.97, 0.995)
INITIAL_DAYS_INFECTED: int = 1  # the ECS counts the seeding day for the first infected

# Quarantine (fixed in the ECS ``_register_systems`` / ``QuarantineSystem``).
QUARANTINE_COMPLIANCE: float = 0.8
QUARANTINE_DURATION: int = 14
QUARANTINE_EFFECT: float = 0.9  # up to 90 % reduction of mobility / transmission

# Dynamic rewiring uses a fixed dispersion in the ECS (the initial network uses ``dispersion``).
REWIRING_DISPERSION: float = 0.5

# Contact network: strengths of the random network are U(0.1, 1).
RANDOM_NETWORK_STRENGTH_RANGE: tuple[float, float] = (0.1, 1.0)


def health_counts(n_agents: int, initial_infected: int,
                  proportions: Optional[Mapping[State, float]] = None) -> dict[State, int]:
    """Number of people in each health state at the start.

    Without ``proportions`` the population is ``initial_infected`` infected people and
    everybody else susceptible. With ``proportions`` (a mapping state -> share, summing
    to 1) the counts are the rounded shares, the remainder going to the susceptible.
    """
    if proportions is None:
        return {State.Susceptible: n_agents - initial_infected,
                State.Infected: initial_infected,
                State.Recovered: 0, State.Dead: 0}
    if any(p < 0 for p in proportions.values()) or abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError("initial proportions must be non-negative and sum to 1")
    counts = {state: int(round(proportions.get(state, 0.0) * n_agents)) for state in State}
    others = sum(c for s, c in counts.items() if s is not State.Susceptible)
    if others > n_agents:
        raise ValueError("initial proportions give more people than n_agents")
    counts[State.Susceptible] = n_agents - others
    return counts
