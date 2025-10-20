# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Direct workflow environments.
"""

import gymnasium as gym

# Added by Maxence Boels
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Quantum-Tracer-Direct-v0",
    entry_point=f"{__name__}.quantum_tracer_env:QuantumTracerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quantum_tracer_env:QuantumTracerEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
