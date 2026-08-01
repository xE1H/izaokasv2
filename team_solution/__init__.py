"""Your solution. This is the half of the repository you edit.

Importing this package registers the task under ``"LituanicaX-Race-Team-v0"``,
which is what ``train.py`` and ``play.py`` (both in here) use by default.

* ``env.py`` — ★ the file you work in: what the policy sees, what it is paid
  for, when an episode ends, and where the cars drive. Plain tensor code, all
  of it yours.
* ``ppo_cfg.py`` — the learning settings: network size, learning rate, batch
  size.
* ``tracks/`` — your own tracks. Drop a walls USD and a centerline CSV in here
  and register them.

The car, the physics and the lap clock come from ``lituanicax_sdk/`` and are
the same for every team — see ``README.md``.
"""

import gymnasium as gym

from . import tracks  # noqa: F401 — importing registers any tracks you added

gym.register(
    id="LituanicaX-Race-Team-v0",
    entry_point=f"{__name__}.env:TeamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:TeamRaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:TeamPPORunnerCfg",
    },
)
