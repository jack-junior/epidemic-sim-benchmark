"""Mesa (object-oriented) implementation of the spatial epidemic model.

This model is the benchmark counterpart of ``epidemic_sim.ecs.simulation.SIREpidemicModel``.
It keeps the same parameters, the same mechanisms, the same order of the
systems inside one step, the same output definitions and -- for the
initialisation -- the same order of random draws, so that with the same seed
the initial population, the initial infections and the initial contact network
are identical to the ECS ones.

Order of one step (highest ECS priority first):

    1. movement                       (ECS priority 100)
    2. contact-network rewiring       (95, similarity network only, every ``rewire_every`` steps)
    3. spatial transmission           (90)
    4. network transmission           (80)
    5. infection resolution           (70, when a "new" hazard-based system is used)
    6. disease progression            (60)
    7. quarantine                     (60, optional, after progression)
    8. data collection

Intentional differences with the ECS code (they do not change the model):
    * the diagnostic printing of the ECS rewiring/network builders is omitted;
    * agents are Python objects, contacts are references to ``Person`` objects.

Behaviours that are copied as-is from the ECS code, including two that the
author may want to review:
    * the rewiring uses a fixed dispersion ``k = 0.5`` while the initial
      network uses the ``dispersion`` parameter;
    * contacts are directed (i chooses j; j does not necessarily choose i).
"""

import math
from collections import defaultdict
from typing import Optional

import mesa

from ..randomness import RandomStreams
from .agents import DEAD, INFECTED, RECOVERED, SUSCEPTIBLE, Person

QUARANTINE_COMPLIANCE = 0.8  # fixed in the ECS ``_register_systems``
REWIRING_DISPERSION = 0.5    # fixed in the ECS ``NetworkRewiringystem``


class MesaSIREpidemicModel(mesa.Model):
    def __init__(self, seed: int, tau: float = 25, alpha=None, n_agents: int = 500,
                 world_size: int = 100, initial_infected: int = 5,
                 average_contacts: int = 10, beta_spatial: float = 0.10,
                 beta_network: float = 0.20, enable_quarantine: bool = False,
                 transmission_radius: float = 4.0, spatial_new: Optional[bool] = False,
                 network_new: Optional[bool] = False,
                 space_attribute_similarity: Optional[bool] = False,
                 dt: float = 1.0, dispersion: float = 0.7, rewire_every: int = 18,
                 collect_spatial: bool = True):
        self._validate_parameters(seed, tau, alpha, n_agents, world_size,
                                  initial_infected, average_contacts,
                                  beta_spatial, beta_network,
                                  transmission_radius, dt, dispersion)
        # Mesa draws from its own generator while initialising (``model.rng``,
        # unused here): the model-owned streams below must stay independent of it,
        # otherwise the draw sequence would be shifted w.r.t. the ECS model.
        super().__init__(rng=seed)
        self.streams = RandomStreams.from_seed(seed)  # same construction as the ECS model

        self.n_agents = n_agents
        self.world_size = world_size
        self.initial_infected = initial_infected
        self.average_contacts = average_contacts
        self.beta_spatial = beta_spatial
        self.beta_network = beta_network
        self.enable_quarantine = enable_quarantine
        self.seed = seed
        self.transmission_radius = transmission_radius
        self.dt = dt
        self.alpha = alpha if alpha is not None else 0.20 * world_size
        self.tau = tau
        self.dispersion = dispersion
        self.rewire_every = rewire_every
        self.collect_spatial = collect_spatial
        self.spatial_new = bool(spatial_new)
        self.network_new = bool(network_new)
        self._uses_similarity_network = bool(space_attribute_similarity)
        self._rewire_counter = 0
        self.step_count = 0

        self._populate()
        self.seed_locations: list[tuple[float, float]] = []
        self._initial_infection()
        if self._uses_similarity_network:
            self._space_attribute_similarity_network(self.alpha, self.tau, self.dispersion)
        else:
            self._create_social_network()

        self.time_series_data: defaultdict[str, list] = defaultdict(list)
        self.spatial_location_series_data: defaultdict[str, list] = defaultdict(list)

    @staticmethod
    def _validate_parameters(seed, tau, alpha, n_agents, world_size,
                             initial_infected, average_contacts, beta_spatial,
                             beta_network, transmission_radius, dt, dispersion):
        # Same checks as the ECS model (duplicated on purpose: importing the ECS
        # module here would load esper and pollute the memory of a Mesa-only run).
        if seed is not None and not isinstance(seed, int):
            raise TypeError("seed must be an integer or None")
        if n_agents < 1 or initial_infected < 0 or initial_infected > n_agents:
            raise ValueError("n_agents must be positive and initial_infected must be in [0, n_agents]")
        if world_size <= 0 or average_contacts < 0 or transmission_radius < 0 or dt <= 0:
            raise ValueError("world_size and dt must be positive; contacts and radius cannot be negative")
        if tau <= 0 or dispersion <= 0:
            raise ValueError("tau and dispersion must be positive")
        if alpha is not None and alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= beta_spatial <= 1 or not 0 <= beta_network <= 1:
            raise ValueError("transmission probabilities must be between 0 and 1")

    # ------------------------------------------------------------------
    # Initialisation (same order of random draws as the ECS model)
    # ------------------------------------------------------------------
    def _populate(self):
        rng = self.streams.numpy
        for _ in range(self.n_agents):
            x = rng.uniform(0, self.world_size)
            y = rng.uniform(0, self.world_size)
            age = int(rng.integers(0, 75))
            mobility = rng.uniform(0.5, 20.0)
            immunity = rng.uniform(0, 0.05)
            Person(self, x, y, age, mobility, immunity)

    def _initial_infection(self):
        chosen = self.streams.python.sample(list(self.agents),
                                            min(self.initial_infected, self.n_agents))
        for person in chosen:
            viral_load = self.streams.numpy.uniform(700, 1000)
            recovery_time = max(1, int(self.streams.python.normalvariate(12, 4)))
            person.infect(viral_load, recovery_time, days_infected=1)
            self.seed_locations.append((person.x, person.y))

    def _create_social_network(self):
        """Random network: Poisson number of contacts, strengths U(0.1, 1)."""
        for person in self.agents:
            num_contacts = max(0, int(self.streams.numpy.poisson(self.average_contacts)))
            if num_contacts > 0:
                possible = [other for other in self.agents if other is not person]
                num_contacts = min(num_contacts, len(possible))
                person.contacts = self.streams.python.sample(possible, num_contacts)
                person.contact_strength = [self.streams.python.uniform(0.1, 1.0)
                                           for _ in range(num_contacts)]

    def _draw_similarity_contacts(self, person: Person, num_contacts: int,
                                  alive_only: bool):
        """Draw ``num_contacts`` contacts for ``person`` with probability
        proportional to w_ij = exp(-d_ij/alpha) * exp(-|age_i - age_j|/tau)
        (naive O(N) loop, same algorithm as the ECS code)."""
        candidates = []
        weights = []
        for other in self.agents:
            if other is person:
                continue
            if alive_only and other.state == DEAD:
                continue
            distance = math.sqrt((person.x - other.x) ** 2 + (person.y - other.y) ** 2)
            age_difference = abs(person.age - other.age)
            weights.append(math.exp(-distance / self.alpha) *
                           math.exp(-age_difference / self.tau))
            candidates.append(other)

        total_weight = sum(weights)
        if total_weight == 0:
            return
        probability = [weight / total_weight for weight in weights]
        chosen = self.streams.numpy.choice(len(candidates),
                                           size=min(num_contacts, len(candidates)),
                                           replace=False, p=probability)
        person.contacts = [candidates[i] for i in chosen]
        person.contact_strength = [weights[i] for i in chosen]  # raw similarity = tie strength

    def _space_attribute_similarity_network(self, alpha: float, tau: float,
                                            dispersion: float):
        """Spatial + age homophily network (initial construction)."""
        for person in self.agents:
            num_contacts = max(0, int(self.streams.numpy.negative_binomial(
                dispersion, dispersion / (dispersion + self.average_contacts))))
            if num_contacts == 0:
                continue
            self._draw_similarity_contacts(person, num_contacts, alive_only=False)

    # ------------------------------------------------------------------
    # Systems
    # ------------------------------------------------------------------
    def _rewire_network(self):
        """Dynamic rewiring every ``rewire_every`` steps (ECS ``NetworkRewiringystem``)."""
        self._rewire_counter += 1
        if self._rewire_counter % self.rewire_every != 0:
            return
        k = REWIRING_DISPERSION
        for person in self.agents:
            if person.state == DEAD:
                continue
            person.contacts = []
            person.contact_strength = []
            num_contacts = max(0, int(self.streams.numpy.negative_binomial(
                k, k / (k + self.average_contacts))))
            if num_contacts == 0:
                continue
            self._draw_similarity_contacts(person, num_contacts, alive_only=True)

    def _draw_new_infection(self):
        """(viral_load, recovery_time) of a new infection, same draw order as the ECS."""
        rng = self.streams.python
        viral_load = rng.uniform(500, 1000)
        recovery_time = max(1, int(rng.normalvariate(12, 4)))
        return viral_load, recovery_time

    def _spatial_transmission_hazard(self):
        """Hazard-based spatial transmission (ECS ``SpatialTransmissionSystemNew``)."""
        infectious = [a for a in self.agents
                      if a.state == INFECTED and a.infectious and not a.quarantined]
        if not infectious:
            return
        radius = self.transmission_radius
        base = self.beta_spatial
        for susceptible in self.agents:
            if susceptible.state != SUSCEPTIBLE:
                continue
            accumulated = 0.0
            for source in infectious:
                dx = source.x - susceptible.x
                dy = source.y - susceptible.y
                if math.sqrt(dx ** 2 + dy ** 2) <= radius:
                    accumulated += (base * source.viral_load / 1000) * (1 - susceptible.immunity)
            if accumulated > 0:
                susceptible.hazard += accumulated

    def _spatial_transmission_immediate(self):
        """Sequential spatial transmission (ECS ``SpatialTransmissionSystem``)."""
        infected = [a for a in self.agents if a.state == INFECTED]
        susceptible = [a for a in self.agents if a.state == SUSCEPTIBLE]
        rng = self.streams.python
        for source in infected:
            if not source.infectious or source.quarantined:
                continue
            for target in susceptible:
                dx = source.x - target.x
                dy = source.y - target.y
                if (dx ** 2 + dy ** 2) ** 0.5 <= self.transmission_radius:
                    probability = (self.beta_spatial * source.viral_load / 1000) * (1 - target.immunity)
                    if rng.random() < probability and target.state == SUSCEPTIBLE:
                        target.infect(*self._draw_new_infection())

    def _network_transmission_hazard(self):
        """Hazard-based network transmission (ECS ``NetworkTransmissionSystemNew``)."""
        for source in self.agents:
            if source.state != INFECTED or not source.infectious:
                continue
            adjusted = self.beta_network * source.transmission_modifier()
            for contact, strength in zip(source.contacts, source.contact_strength):
                if contact.state != SUSCEPTIBLE:
                    continue
                hazard = adjusted * strength * (1 - contact.immunity) * source.viral_load / 1000
                if hazard > 0:
                    contact.hazard += hazard

    def _network_transmission_immediate(self):
        """Sequential network transmission (ECS ``NetworkTransmissionSystem``)."""
        sources = [a for a in self.agents if a.state == INFECTED and a.contacts]
        rng = self.streams.python
        for source in sources:
            if not source.infectious:
                continue
            adjusted = self.beta_network * source.transmission_modifier()
            for contact, strength in zip(source.contacts, source.contact_strength):
                if contact.state == SUSCEPTIBLE:
                    probability = adjusted * strength * (1 - contact.immunity) * source.viral_load / 1000
                    if rng.random() < probability:
                        contact.infect(*self._draw_new_infection())

    def _resolve_infections(self):
        """Hazard -> probability p = 1 - exp(-hazard * dt) (ECS ``InfectionResolutionSystem``)."""
        rng = self.streams.python
        for person in self.agents:
            if person.state != SUSCEPTIBLE or person.hazard <= 0:
                continue
            probability = 1 - math.exp(-person.hazard * self.dt)
            if rng.random() < probability:
                person.infect(*self._draw_new_infection())
            person.hazard = 0.0

    # ------------------------------------------------------------------
    # One step / run
    # ------------------------------------------------------------------
    def step(self):
        self.agents.do("move")
        if self._uses_similarity_network:
            self._rewire_network()
        if self.spatial_new:
            self._spatial_transmission_hazard()
        else:
            self._spatial_transmission_immediate()
        if self.network_new:
            self._network_transmission_hazard()
        else:
            self._network_transmission_immediate()
        if self.spatial_new or self.network_new:
            self._resolve_infections()
        self.agents.do("progress_disease")
        if self.enable_quarantine:
            self.agents.do("try_quarantine", QUARANTINE_COMPLIANCE)
            self.agents.do("tick_quarantine")

        self._collect_data()
        if self.collect_spatial:
            self.get_spatial_data()
        self.step_count += 1

    def _count(self, state: str) -> int:
        return sum(1 for a in self.agents if a.state == state)

    def _collect_data(self):
        """Counts of each health state (same definition as the ECS model)."""
        self.time_series_data["time"].append(self.step_count)
        self.time_series_data["susceptible"].append(self._count(SUSCEPTIBLE))
        self.time_series_data["infected"].append(self._count(INFECTED))
        self.time_series_data["recovered"].append(self._count(RECOVERED))
        self.time_series_data["death"].append(self._count(DEAD))
        return self.time_series_data

    def get_spatial_data(self):
        """(step, x, y) of every agent, by health state."""
        names = {SUSCEPTIBLE: "susceptible", INFECTED: "infected",
                 RECOVERED: "recovered", DEAD: "death"}
        for person in self.agents:
            self.spatial_location_series_data[names[person.state]].append(
                (self.step_count, person.x, person.y))
        return self.spatial_location_series_data

    def run(self, max_steps: int):
        """Run until ``max_steps`` or until nobody is infected (same rule as the ECS)."""
        for _ in range(max_steps):
            self.step()
            if self._count(INFECTED) == 0:
                print(f"Simulation ended at step {self.step_count} - no more infected entities.")
                break

    def clean_up(self):
        """Nothing to release: the model owns all its state (no global world)."""
