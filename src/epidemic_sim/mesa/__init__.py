"""Mesa implementation of the epidemic model."""

from .agents import Person
from .model import MesaSIREpidemicModel

__all__ = ["MesaSIREpidemicModel", "Person"]
