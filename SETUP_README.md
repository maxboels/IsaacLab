# Isaac Lab Setup Guide

This document outlines the setup process for the Isaac Lab environment with pip installation method.

## System Information

- **OS**: Ubuntu 24.04.3 LTS (Noble Numbat)
- **GLIBC Version**: 2.39 (Required: 2.35+)
- **Python**: 3.11.14
- **CUDA**: 12.8
- **GPU**: NVIDIA GeForce RTX 4060 Laptop
- **Isaac Sim Version**: 5.0.0
- **Isaac Lab Version**: 0.47.1

## Installation Steps Completed

### 1. Virtual Environment Setup

Created a virtual environment using UV:

```bash
cd /home/maxboels/projects/IsaacLab
uv venv env_isaaclab
source env_isaaclab/bin/activate
```

**Note**: Deactivate conda base environment to avoid conflicts:
```bash
conda deactivate
```

### 2. Isaac Sim Installation (via pip)

Installed Isaac Sim 5.0.0 with all extensions:

```bash
# Install PyTorch with CUDA 12.8 support
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install Isaac Sim
uv pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
```

**Verification**: Isaac Sim was successfully tested in both headless and GUI modes.

### 3. Isaac Lab Installation

Cloned and installed Isaac Lab:

```bash
# Clone the repository
cd /home/maxboels/projects/IsaacLab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Install system dependencies (Linux only)
sudo apt install cmake build-essential

# Install Isaac Lab extensions and learning frameworks
./isaaclab.sh --install
```

This installed:
- Core Isaac Lab extension (`isaaclab`)
- Assets extension (`isaaclab_assets`)
- Mimic extension (`isaaclab_mimic`)
- RL extension (`isaaclab_rl`)
- Tasks extension (`isaaclab_tasks`)
- Learning frameworks: rl_games, rsl_rl, skrl, stable-baselines3, robomimic

### 4. Git Repository Setup

Forked the official Isaac Lab repository and configured remotes:

```bash
# Update origin to point to personal fork
git remote set-url origin https://github.com/maxboels/IsaacLab.git

# Add upstream to track official Isaac Lab repo
git remote add upstream https://github.com/isaac-sim/IsaacLab.git

# Verify remotes
git remote -v
```

**Result**:
- `origin`: Your fork (https://github.com/maxboels/IsaacLab.git)
- `upstream`: Official repo (https://github.com/isaac-sim/IsaacLab.git)

## Directory Structure

```
/home/maxboels/projects/IsaacLab/
├── docs/                    # Documentation files
├── env_isaaclab/           # Virtual environment
└── IsaacLab/               # Main Isaac Lab repository
    ├── source/
    │   ├── isaaclab/       # Core library
    │   ├── isaaclab_assets/
    │   │   └── isaaclab_assets/
    │   │       ├── robots/  # Pre-built robot configurations
    │   │       ├── sensors/ # Sensor configurations
    │   │       └── direct/  # Custom assets location
    │   ├── isaaclab_mimic/  # Robomimic integration
    │   ├── isaaclab_rl/     # RL framework integrations
    │   └── isaaclab_tasks/  # Pre-built tasks/environments
    ├── scripts/             # Utility scripts
    ├── apps/                # Application configurations
    └── tools/               # Development tools
```

## Daily Workflow

### Activating the Environment

Always activate the virtual environment before working:

```bash
cd /home/maxboels/projects/IsaacLab
source env_isaaclab/bin/activate
```

### Running Scripts

Use the Isaac Lab helper script:

```bash
cd /home/maxboels/projects/IsaacLab/IsaacLab

# Run Python scripts
./isaaclab.sh -p path/to/your_script.py

# Or if virtual environment is activated, use python directly
python path/to/your_script.py
```

### Testing Installation

Verify everything works:

```bash
# Test Isaac Sim in headless mode
python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); app.close()"

# Run a simple Isaac Lab example (once available)
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

## Git Workflow for Custom Development

### Making Changes

```bash
# Create a new branch for your custom robot
git checkout -b feature/my-custom-robot

# Make your changes, then commit
git add .
git commit -m "Add custom robot configuration"

# Push to your fork
git push origin feature/my-custom-robot
```

### Pulling Updates from Official Isaac Lab

```bash
# Fetch updates from official repo
git fetch upstream

# Merge updates into your main branch
git checkout main
git merge upstream/main

# Push updates to your fork
git push origin main
```

## Adding Custom Robots

Custom robots should be added to:
```
source/isaaclab_assets/isaaclab_assets/robots/
```

Follow the existing robot configuration patterns (see `franka.py`, `unitree.py`, etc.)

## Known Issues

### VSCode Warning
When using pip installation, you may see:
```
[WARN] Could not find Isaac Sim VSCode settings
```
This **does not affect functionality** - only VSCode IntelliSense features. The Isaac Lab team is working on a fix.

## Installed Packages

Key packages installed:
- `isaacsim==5.0.0` (with all extensions)
- `torch==2.7.0+cu128`
- `isaaclab==0.47.1`
- `warp-lang==1.9.1`
- `gymnasium==1.2.1`
- `rl-games==1.6.1`
- `rsl-rl-lib==3.0.1`
- `stable-baselines3==2.7.0`
- `skrl==1.4.3`
- `robomimic==0.4.0`

## Useful Commands

```bash
# List installed packages
uv pip list

# Update Isaac Lab extensions
./isaaclab.sh --install

# Format code
./isaaclab.sh --format

# Run tests
./isaaclab.sh --test

# Build documentation
./isaaclab.sh --docs
```

## Resources

- **Isaac Lab Docs**: https://isaac-sim.github.io/IsaacLab/
- **Isaac Sim Forums**: https://docs.isaacsim.omniverse.nvidia.com/latest/common/feedback.html
- **Your Fork**: https://github.com/maxboels/IsaacLab
- **Official Repo**: https://github.com/isaac-sim/IsaacLab

## Next Steps

1. Explore example robots in `source/isaaclab_assets/isaaclab_assets/robots/`
2. Review example tasks in `source/isaaclab_tasks/`
3. Create your custom robot configuration
4. Develop training environments for your robot
5. Train using the integrated RL frameworks

---

**Setup completed on**: October 20, 2025
**Setup by**: maxboels
