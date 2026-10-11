"""Centralized parameter validation for the SIR epidemic model."""

from .config import SimulationConfig
from .disease_model import DiseaseModelConfig


def validate_simulation_config(config: SimulationConfig) -> None:
    """Validate simulation sizing and execution parameters."""
    if config.n_agents < 1:
        raise ValueError(f"n_agents must be >= 1, got {config.n_agents}")
    if not 0 <= config.initial_infected <= config.n_agents:
        raise ValueError(
            f"initial_infected must be in [0, {config.n_agents}], "
            f"got {config.initial_infected}"
        )
    if config.world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {config.world_size}")
    if config.average_contacts < 0:
        raise ValueError(
            f"average_contacts must be >= 0, got {config.average_contacts}"
        )
    if config.max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {config.max_steps}")
    if config.seed is not None and not isinstance(config.seed, int):
        raise TypeError(f"seed must be int or None, got {type(config.seed)}")


def validate_disease_model_config(config: DiseaseModelConfig) -> None:
    """Explicitly revalidate a disease-model configuration."""
    config.__post_init__()


def validate_all(
    sim_config: SimulationConfig,
    disease_config: DiseaseModelConfig,
) -> None:
    """Validate both simulation and disease-model configurations."""
    validate_simulation_config(sim_config)
    validate_disease_model_config(disease_config)
