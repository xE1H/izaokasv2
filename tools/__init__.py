"""Development tooling for the optimized-controller track.

Nothing in here is part of a submission. The two halves are split by whether
they need a running simulator:

* **Isaac-free**, and therefore unit-tested on any machine — :mod:`tools.geometry`
  (track maths) and :mod:`tools.profile` (speed profiles, racing lines).
* **Isaac-dependent**, and only runnable on a GPU box with Isaac Sim —
  :mod:`tools.harness`, :mod:`tools.evaluate`, :mod:`tools.probe`,
  :mod:`tools.determinism`.

The split mirrors the SDK's own: ``lituanicax_sdk.timing``, ``dynamics`` and
``track`` are importable without the simulator and tested directly, while
``env`` and ``scene`` are not.
"""
