"""Disease model constants and parameters.

These parameters define the SIR epidemic model semantics across the ECS and
Mesa implementations. Changing a value here updates both implementations.
"""

from dataclasses import dataclass
from enum import Enum


# Population demographics (drawn at initialization)
AGE_RANGE: tuple[int, int] = (0, 75)
MOBILITY_RANGE: tuple[float, float] = (0.5, 20.0)
IMMUNITY_RANGE: tuple[float, float] = (0.0, 0.05)
IMMUNITY_AFTER_RECOVERY: float = 0.9

# Disease progression (infection, recovery, death)
INITIAL_VIRAL_LOAD_RANGE: tuple[float, float] = (700.0, 1000.0)
NEW_VIRAL_LOAD_RANGE: tuple[float, float] = (500.0, 1000.0)
RECOVERY_TIME_MEAN: float = 12.0
RECOVERY_TIME_SD: float = 4.0
RECOVERY_PROBABILITY_RANGE: tuple[float, float] = (0.97, 0.995)
INITIAL_DAYS_INFECTED: int = 1

# Quarantine (fixed across the model)
QUARANTINE_COMPLIANCE: float = 0.8
QUARANTINE_DURATION: int = 14
QUARANTINE_EFFECT: float = 0.9

# Contact network
RANDOM_NETWORK_STRENGTH_RANGE: tuple[float, float] = (0.1, 1.0)
REWIRING_DISPERSION: float = 0.5


class HealthState(Enum):
    """Epidemiological state of a person."""

    SUSCEPTIBLE = "susceptible"
    INFECTED = "infected"
    RECOVERED = "recovered"
    DEAD = "death"


@dataclass(frozen=True)
class DiseaseModelConfig:
    """Immutable specification of disease and transmission parameters."""

    beta_spatial: float = 0.10
    beta_network: float = 0.20
    transmission_radius: float = 4.0
    dt: float = 1.0
    alpha: float | None = None
    tau: float = 25.0
    dispersion: float = 0.7
    spatial_new: bool = False
    network_new: bool = False
    space_attribute_similarity: bool = False
    enable_quarantine: bool = False

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if not 0 <= self.beta_spatial <= 1:
            raise ValueError(
                f"beta_spatial must be in [0, 1], got {self.beta_spatial}"
            )
        if not 0 <= self.beta_network <= 1:
            raise ValueError(
                f"beta_network must be in [0, 1], got {self.beta_network}"
            )
        if self.transmission_radius < 0:
            raise ValueError(
                "transmission_radius must be >= 0, "
                f"got {self.transmission_radius}"
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.alpha is not None and self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.dispersion <= 0:
            raise ValueError(f"dispersion must be > 0, got {self.dispersion}")
