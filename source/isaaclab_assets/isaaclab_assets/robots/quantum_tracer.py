# Copyright Maxence Boels

"""Configuration for the FTX Quantum Tracer RC car.

The following configurations are available:

* :obj:`QUANTUM_TRACER_CFG`: FTX Quantum Tracer 1/10 scale RC truggy

Reference: https://www.ftxrc.com/
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Get absolute path to your USD file
# Assuming you'll place it in: isaac_lab/source/isaaclab_tasks/isaaclab_tasks/direct/quantum_tracer/assets/
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
USD_PATH = os.path.join(
    WORKSPACE_ROOT, 
    "source", 
    "isaaclab_tasks", 
    "isaaclab_tasks", 
    "direct", 
    "quantum_tracer", 
    "assets",
    "quantum_tracer.usd"
)

##
# Configuration
##

QUANTUM_TRACER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=15.0,  # ~50 km/h for 1/10 scale RC car
            max_angular_velocity=50.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),  # Start 15cm above ground
        joint_pos={
            # Rear wheels (drive wheels) - start stationary
            "rear_left_wheel_joint": 0.0,
            "rear_right_wheel_joint": 0.0,
            # Front wheels (for rolling) - start stationary
            "front_left_wheel_joint": 0.0,
            "front_right_wheel_joint": 0.0,
            # Steering joints - start centered
            "front_left_steering_joint": 0.0,
            "front_right_steering_joint": 0.0,
        },
    ),
    actuators={
        # Drive motor (ESC) - velocity control on rear wheels
        # This mimics how a real RC ESC controls the brushless motor
        "drive_motor": ImplicitActuatorCfg(
            joint_names_expr=["rear_.*_wheel_joint"],
            effort_limit=10.0,  # Nm - adjust based on your car's motor specs
            velocity_limit=200.0,  # rad/s - allows for high speed
            stiffness=0.0,  # No position stiffness for velocity control
            damping=0.5,  # Light damping for smooth velocity tracking
        ),
        # Steering servo - position control on front steering joints
        # This mimics how a real RC servo controls steering angle
        "steering_servo": ImplicitActuatorCfg(
            joint_names_expr=["front_.*_steering_joint"],
            effort_limit=5.0,  # Nm - typical for RC servo
            velocity_limit=20.0,  # rad/s - fast servo response
            stiffness=100.0,  # Stiff for precise position control
            damping=10.0,  # Damping for servo-like behavior
        ),
        # Front wheels (passive rolling) - minimal control
        "front_wheels": ImplicitActuatorCfg(
            joint_names_expr=["front_.*_wheel_joint"],
            effort_limit=0.1,  # Very low - these just roll
            velocity_limit=200.0,
            stiffness=0.0,
            damping=0.1,  # Light damping for realistic rolling
        ),
    },
)
"""Configuration for the FTX Quantum Tracer RC car with realistic control.

Notes:
    - Drive: Velocity control on rear wheels (like ESC → brushless motor)
    - Steering: Position control on front steering joints (like servo)
    - Front wheels: Passive rolling with light damping
    
    This configuration is designed for sim-to-real transfer where:
    - Policy outputs: [steering_angle, throttle] in [-1, 1]
    - steering_angle maps to servo position (-30° to +30°)
    - throttle maps to motor velocity (reverse to forward)
"""
