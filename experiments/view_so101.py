import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
from robot_arm_plays_chess.environments.so101_chess.env import SO101Env

# Create the environment
env = SO101Env(randomize={})

# reset() causes the MuJoCo world to be built
obs, info = env.reset(seed=0)

# Get the actual MuJoCo world assembled by env.py
world = env._ensure_world()

print("Bodies:", world.model.nbody)
print("Joints:", world.model.njnt)
print("Actuators:", world.model.nu)

print("\nJoint state:")
print(obs["agent_pos"])


with mujoco.viewer.launch_passive(world.model, world.data) as viewer:

    while viewer.is_running():

        # Hold the robot at its current position
        action = obs["agent_pos"].copy()

        obs, reward, terminated, truncated, info = env.step(action)

        viewer.sync()

        time.sleep(1 / env.fps)


env.close()
