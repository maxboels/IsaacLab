# Copyright Maxence Boels

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

##
# Configuration
##

WAYPOINT_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/Waypoints",
    markers={
        "current": sim_utils.SphereCfg(
            radius=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red for current target
                metallic=0.2,
            ),
        ),
        "future": sim_utils.SphereCfg(
            radius=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),  # Green for future waypoints
                metallic=0.2,
            ),
        ),
    }
)