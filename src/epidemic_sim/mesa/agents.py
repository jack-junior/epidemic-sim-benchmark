"""Object-oriented agent of the Mesa implementation.

One ``Person`` object holds the whole state of one individual (position,
demographics, health state, infection, quarantine, contacts) and owns the
per-agent behaviour (movement, disease progression, quarantine).  This is the
object-oriented counterpart of the ECS design, where the same information is
split across components (``Location``, ``Demographics``, ``Susceptible``,
``Infected``, ``Recovered``, ``Dead``, ``ContactNetwork``, ``Quarantined``,
``InfectionHazard``) and processed by systems.

The numerical rules are copied from the ECS systems so that both
implementations share exactly the same model semantics.
"""

import mesa

SUSCEPTIBLE = "S"
INFECTED = "I"
RECOVERED = "R"
DEAD = "D"

# Quarantine constants copied from the ECS ``QuarantineSystem``.
QUARANTINE_DURATION = 14
QUARANTINE_EFFECT = 0.9  # up to 90 % reduction of mobility / transmission


class Person(mesa.Agent):
    """One individual of the population."""

    def __init__(self, model, x: float, y: float, age: int,
                 mobility: float, immunity: float):
        super().__init__(model)
        # Location / Demographics / Susceptible components
        self.x = x
        self.y = y
        self.age = age
        self.mobility = mobility
        self.immunity = immunity

        # Health state (replaces the Susceptible/Infected/Recovered/Dead components)
        self.state = SUSCEPTIBLE

        # Infected component
        self.viral_load = 0.0
        self.days_infected = 0
        self.infectious = False
        self.recovery_time = 0

        # InfectionHazard component (0.0 == no hazard component)
        self.hazard = 0.0

        # ContactNetwork component (contacts are references to other Person objects)
        self.contacts: list["Person"] = []
        self.contact_strength: list[float] = []

        # Quarantined component
        self.quarantined = False
        self.quarantine_compliance = 0.0
        self.quarantine_duration = QUARANTINE_DURATION
        self.days_in_quarantine = 0
        self.original_mobility = 0.0

    # ------------------------------------------------------------------
    # State changes
    # ------------------------------------------------------------------
    def infect(self, viral_load: float, recovery_time: int, days_infected: int = 0):
        """Susceptible -> Infected."""
        self.state = INFECTED
        self.viral_load = viral_load
        self.days_infected = days_infected
        self.infectious = True
        self.recovery_time = recovery_time

    def transmission_modifier(self) -> float:
        """Reduction of the network transmission caused by quarantine."""
        if self.quarantined and self.quarantine_compliance > 0:
            return 1.0 - self.quarantine_compliance * QUARANTINE_EFFECT
        return 1.0

    # ------------------------------------------------------------------
    # Behaviours called by the model through ``model.agents.do(...)``
    # ------------------------------------------------------------------
    def move(self):
        """Random walk bounded to the square world (also drawn for the dead,
        whose mobility is 0, exactly like the ECS ``MovementSystem``)."""
        rng = self.model.streams.python
        mobility = self.mobility
        dx = rng.uniform(-mobility, mobility)
        dy = rng.uniform(-mobility, mobility)
        size = self.model.world_size
        self.x = max(0, min(size, self.x + dx))
        self.y = max(0, min(size, self.y + dy))

    def progress_disease(self):
        """Infected: count days, then recover or die (ECS ``DiseaseProgressionSystem``)."""
        if self.state != INFECTED:
            return
        rng = self.model.streams.python
        self.days_infected += 1
        if self.days_infected >= self.recovery_time:
            draw = rng.random()
            recover_probability = rng.uniform(0.97, 0.995)
            if draw < recover_probability:
                self.state = RECOVERED
                self.immunity = 0.9
            else:
                self.state = DEAD
                self.mobility = 0.0
            # A recovered (not dead) agent leaves quarantine at once.
            if self.quarantined and self.state == RECOVERED:
                self.mobility = self.original_mobility
                self.quarantined = False

    def try_quarantine(self, compliance: float):
        """Infectious agents enter quarantine with probability ``compliance``."""
        if self.state == INFECTED and self.infectious and not self.quarantined:
            if self.model.streams.python.random() < compliance:
                self.quarantined = True
                self.quarantine_compliance = compliance
                self.quarantine_duration = QUARANTINE_DURATION
                self.days_in_quarantine = 0
                self.original_mobility = self.mobility
                self.mobility *= (1 - compliance * QUARANTINE_EFFECT)

    def tick_quarantine(self):
        """One more day in quarantine; release when the duration is over."""
        if not self.quarantined:
            return
        self.days_in_quarantine += 1
        if self.days_in_quarantine >= self.quarantine_duration:
            if self.state != DEAD:
                self.mobility = self.original_mobility
            self.quarantined = False
