"""The optimized deterministic controller — an instrument and a teacher.

**This is not the submission and never will be.** A submission is a neural
network checkpoint; a hand-written controller does not pass the SDK's own
interface, and could not be verified by re-running it. The controller exists for
two other reasons:

1. **Instrument.** It measures the achievable lap time on this track under real
   physics — a number produced by a car actually completing a lap under the
   actual crash rules, rather than by a point-mass estimate.
2. **Teacher.** Its optimized line and speed profile become reference
   observations, and its actions become demonstrations, for a distilled policy.

Because of (2), the controller may only read quantities the SDK exposes to a
policy. That is enforced structurally rather than by discipline: it is called
with a :class:`~lituanicax_sdk.state.CarState` and nothing else, which is exactly
the surface a policy gets. See :class:`teacher.controller.Controller`.

Layout:

* :mod:`teacher.params` — the ~70 parameters, their bounds, and normalization.
* :mod:`teacher.controller` — the control law, vectorized over thousands of cars.
* :mod:`teacher.warmstart` — the starting parameter vector.
* :mod:`teacher.optimize` — CMA-ES against the simulator. Needs Isaac Sim.
"""
