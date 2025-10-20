# Copyright Maxence Boels

from __future__ import annotations

import torch
from collections.abc import Sequence
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab_assets.robots.quantum_tracer import QUANTUM_TRACER_CFG
from .waypoint import WAYPOINT_CFG


@configclass
class QuantumTracerEnvCfg(DirectRLEnvCfg):
    """Configuration for the Quantum Tracer navigation environment."""
    
    # Environment settings
    decimation = 4  # Control frequency: 60Hz / 4 = 15Hz
    episode_length_s = 30.0
    
    # Action and observation spaces
    action_space = 2  # [steering, throttle]
    observation_space = 10  # See _get_observations for details
    state_space = 0
    
    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,  # 60Hz physics
        render_interval=decimation
    )
    
    # Robot configuration
    robot_cfg: ArticulationCfg = QUANTUM_TRACER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    
    # Waypoint visualization
    waypoint_cfg = WAYPOINT_CFG
    
    # Joint names (adjust these based on your URDF/USD)
    drive_joint_names = [
        "rear_left_wheel_joint",
        "rear_right_wheel_joint"
    ]
    steering_joint_names = [
        "front_left_steering_joint",
        "front_right_steering_joint",
    ]
    
    # Action scaling
    max_steering_angle = 0.524  # 30 degrees in radians
    max_wheel_velocity = 150.0  # rad/s
    
    # Scene
    env_spacing = 32.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=env_spacing,
        replicate_physics=True
    )
    
    # Waypoint course parameters
    num_waypoints = 10
    course_length = 25.0  # meters
    course_width = 5.0  # meters
    waypoint_radius = 0.5  # meters - how close to be considered "reached"
    
    # Reward weights
    rew_position_progress = 1.0
    rew_heading_alignment = 0.1
    rew_waypoint_reached = 10.0
    rew_course_completion = 100.0
    rew_action_smoothness = -0.01
    rew_speed_penalty = -0.001
    
    # Termination conditions
    max_position_error = 10.0  # meters off course
    min_progress_timeout = 5.0  # seconds without reaching next waypoint


class QuantumTracerEnv(DirectRLEnv):
    """
    RC car navigation environment using waypoint following.
    
    The agent must navigate through a series of waypoints by controlling:
    - Steering angle (servo-like position control)
    - Throttle (ESC-like velocity control)
    """
    
    cfg: QuantumTracerEnvCfg

    def __init__(self, cfg: QuantumTracerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Find joint indices
        self._drive_joint_idx, _ = self.car.find_joints(self.cfg.drive_joint_names)
        self._steering_joint_idx, _ = self.car.find_joints(self.cfg.steering_joint_names)
        
        # Action storage
        self._throttle_action = torch.zeros(self.num_envs, len(self._drive_joint_idx), 
                                           device=self.device, dtype=torch.float32)
        self._steering_action = torch.zeros(self.num_envs, len(self._steering_joint_idx), 
                                           device=self.device, dtype=torch.float32)
        
        # Waypoint tracking
        self._waypoint_positions = torch.zeros((self.num_envs, self.cfg.num_waypoints, 2), 
                                              device=self.device, dtype=torch.float32)
        self._current_waypoint_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._waypoints_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # For reward calculation
        self._previous_distance_to_waypoint = torch.zeros(self.num_envs, device=self.device)
        self._time_since_last_waypoint = torch.zeros(self.num_envs, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 2, device=self.device)
        
        # Markers for visualization
        self._marker_positions = torch.zeros((self.num_envs, self.cfg.num_waypoints, 3), 
                                            device=self.device, dtype=torch.float32)

    def _setup_scene(self):
        """Setup the scene with car, ground plane, and markers."""
        
        # Spawn ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(1000.0, 1000.0),
                color=(0.2, 0.2, 0.2),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=0.8,  # Asphalt-like friction
                    dynamic_friction=0.7,
                    restitution=0.01,  # Low bounce
                ),
            ),
        )
        
        # Create car articulation
        self.car = Articulation(self.cfg.robot_cfg)
        
        # Create waypoint markers
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        
        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["car"] = self.car
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step.
        
        Actions are expected in range [-1, 1]:
        - actions[:, 0]: steering angle (-1 = full left, +1 = full right)
        - actions[:, 1]: throttle (-1 = full reverse, +1 = full forward)
        """
        # Scale steering to physical angle limits
        steering_angle = actions[:, 0] * self.cfg.max_steering_angle
        self._steering_action = steering_angle.unsqueeze(1).repeat(1, len(self._steering_joint_idx))
        
        # Scale throttle to wheel velocity
        wheel_velocity = actions[:, 1] * self.cfg.max_wheel_velocity
        self._throttle_action = wheel_velocity.unsqueeze(1).repeat(1, len(self._drive_joint_idx))
        
        # Store for smoothness penalty
        self._previous_actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply computed actions to the car."""
        # Set steering position targets (servo-like)
        self.car.set_joint_position_target(
            self._steering_action, 
            joint_ids=self._steering_joint_idx
        )
        
        # Set drive velocity targets (ESC-like)
        self.car.set_joint_velocity_target(
            self._throttle_action,
            joint_ids=self._drive_joint_idx
        )

    def _get_observations(self) -> dict:
        """Compute observations for the policy.
        
        Returns:
            Dictionary with 'policy' key containing observation tensor of shape (num_envs, 10):
            - [0]: Distance to current waypoint (normalized)
            - [1-2]: Direction to waypoint in car's frame (x, y)
            - [3-4]: Car velocity in car's frame (forward, lateral)
            - [5]: Car angular velocity (yaw rate)
            - [6]: Heading error (cos)
            - [7]: Heading error (sin)
            - [8-9]: Previous actions (steering, throttle)
        """
        # Get current waypoint
        current_waypoint = self._waypoint_positions[
            torch.arange(self.num_envs, device=self.device),
            self._current_waypoint_idx
        ]
        
        # Vector to waypoint in world frame
        to_waypoint_world = current_waypoint - self.car.data.root_pos_w[:, :2]
        distance_to_waypoint = torch.norm(to_waypoint_world, dim=1)
        
        # Transform to car's local frame
        heading = self.car.data.heading_w
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        to_waypoint_local = torch.stack([
            to_waypoint_world[:, 0] * cos_h + to_waypoint_world[:, 1] * sin_h,
            -to_waypoint_world[:, 0] * sin_h + to_waypoint_world[:, 1] * cos_h
        ], dim=1)
        
        # Heading error to waypoint
        target_heading = torch.atan2(to_waypoint_world[:, 1], to_waypoint_world[:, 0])
        heading_error = torch.atan2(
            torch.sin(target_heading - heading),
            torch.cos(target_heading - heading)
        )
        
        # Normalize distance (helps with learning)
        distance_normalized = distance_to_waypoint / self.cfg.course_length
        
        # Compose observation
        obs = torch.cat([
            distance_normalized.unsqueeze(1),
            to_waypoint_local / self.cfg.course_length,  # Normalized direction
            self.car.data.root_lin_vel_b[:, :2],  # Forward and lateral velocity
            self.car.data.root_ang_vel_w[:, 2:3],  # Yaw rate
            torch.cos(heading_error).unsqueeze(1),
            torch.sin(heading_error).unsqueeze(1),
            self._previous_actions,  # Steering and throttle
        ], dim=1)
        
        # Store for reward computation
        self._previous_distance_to_waypoint = distance_to_waypoint
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on navigation progress."""
        
        # Current state
        current_waypoint = self._waypoint_positions[
            torch.arange(self.num_envs, device=self.device),
            self._current_waypoint_idx
        ]
        to_waypoint = current_waypoint - self.car.data.root_pos_w[:, :2]
        distance_to_waypoint = torch.norm(to_waypoint, dim=1)
        
        # Progress reward: moved closer to waypoint
        progress = self._previous_distance_to_waypoint - distance_to_waypoint
        progress_reward = progress * self.cfg.rew_position_progress
        
        # Heading alignment reward
        target_heading = torch.atan2(to_waypoint[:, 1], to_waypoint[:, 0])
        heading_error = torch.abs(torch.atan2(
            torch.sin(target_heading - self.car.data.heading_w),
            torch.cos(target_heading - self.car.data.heading_w)
        ))
        heading_reward = torch.exp(-heading_error) * self.cfg.rew_heading_alignment
        
        # Waypoint reached bonus
        waypoint_reached = (distance_to_waypoint < self.cfg.waypoint_radius).float()
        waypoint_bonus = waypoint_reached * self.cfg.rew_waypoint_reached
        
        # Update waypoint tracking
        self._current_waypoint_idx = torch.where(
            waypoint_reached.bool(),
            (self._current_waypoint_idx + 1) % self.cfg.num_waypoints,
            self._current_waypoint_idx
        )
        self._waypoints_reached += waypoint_reached.long()
        self._time_since_last_waypoint = torch.where(
            waypoint_reached.bool(),
            torch.zeros_like(self._time_since_last_waypoint),
            self._time_since_last_waypoint + self.cfg.sim.dt * self.cfg.decimation
        )
        
        # Course completion bonus
        course_complete = (self._waypoints_reached >= self.cfg.num_waypoints).float()
        completion_bonus = course_complete * self.cfg.rew_course_completion
        
        # Action smoothness penalty (discourage jerky control)
        # This will be computed on next step when we have new actions
        
        # Speed penalty (discourage excessive speed)
        speed = torch.norm(self.car.data.root_lin_vel_w[:, :2], dim=1)
        speed_penalty = speed ** 2 * self.cfg.rew_speed_penalty
        
        # Update marker visualization
        one_hot = torch.nn.functional.one_hot(
            self._current_waypoint_idx,
            num_classes=self.cfg.num_waypoints
        )
        self.waypoints.visualize(marker_indices=one_hot.view(-1).tolist())
        
        # Total reward
        total_reward = (
            progress_reward +
            heading_reward +
            waypoint_bonus +
            completion_bonus +
            speed_penalty
        )
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination and timeout conditions."""
        
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Off course (too far from current waypoint)
        current_waypoint = self._waypoint_positions[
            torch.arange(self.num_envs, device=self.device),
            self._current_waypoint_idx
        ]
        distance = torch.norm(
            current_waypoint - self.car.data.root_pos_w[:, :2],
            dim=1
        )
        off_course = distance > self.cfg.max_position_error
        
        # No progress timeout
        no_progress = self._time_since_last_waypoint > self.cfg.min_progress_timeout
        
        # Task completion
        task_complete = self._waypoints_reached >= self.cfg.num_waypoints
        
        terminated = off_course | no_progress
        
        return terminated, time_out | task_complete

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self.car._ALL_INDICES
        
        super()._reset_idx(env_ids)
        
        num_reset = len(env_ids)
        
        # Reset car to default state
        default_state = self.car.data.default_root_state[env_ids]
        car_pose = default_state[:, :7]
        car_velocities = default_state[:, 7:]
        joint_pos = self.car.data.default_joint_pos[env_ids]
        joint_vel = self.car.data.default_joint_vel[env_ids]
        
        # Position car at origin of each environment
        car_pose[:, :3] += self.scene.env_origins[env_ids]
        car_pose[:, 2] += 0.15  # Start slightly above ground
        
        # Random initial orientation (±15 degrees)
        angles = (torch.rand(num_reset, device=self.device) - 0.5) * (math.pi / 6)
        car_pose[:, 3] = torch.cos(angles * 0.5)  # w component of quaternion
        car_pose[:, 6] = torch.sin(angles * 0.5)  # z component of quaternion
        
        # Write to simulation
        self.car.write_root_pose_to_sim(car_pose, env_ids)
        self.car.write_root_velocity_to_sim(car_velocities, env_ids)
        self.car.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Generate waypoint course
        self._generate_waypoints(env_ids)
        
        # Reset tracking variables
        self._current_waypoint_idx[env_ids] = 0
        self._waypoints_reached[env_ids] = 0
        self._time_since_last_waypoint[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        # Initialize distance tracking
        first_waypoint = self._waypoint_positions[env_ids, 0]
        self._previous_distance_to_waypoint[env_ids] = torch.norm(
            first_waypoint - car_pose[:, :2],
            dim=1
        )

    def _generate_waypoints(self, env_ids: torch.Tensor):
        """Generate waypoint course for specified environments."""
        num_reset = len(env_ids)
        
        # Generate waypoints along a straight line with random lateral offset
        spacing = self.cfg.course_length / self.cfg.num_waypoints
        
        for i in range(self.cfg.num_waypoints):
            # Forward progress
            x = (i + 1) * spacing - self.cfg.course_length / 2
            # Random lateral position
            y = (torch.rand(num_reset, device=self.device) - 0.5) * self.cfg.course_width
            
            self._waypoint_positions[env_ids, i, 0] = x
            self._waypoint_positions[env_ids, i, 1] = y
        
        # Offset by environment origins
        self._waypoint_positions[env_ids] += self.scene.env_origins[env_ids, :2].unsqueeze(1)
        
        # Update marker positions for visualization
        self._marker_positions[env_ids, :, :2] = self._waypoint_positions[env_ids]
        self._marker_positions[env_ids, :, 2] = 0.2  # Height above ground
        
        # Visualize all waypoints
        visualize_pos = self._marker_positions.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)