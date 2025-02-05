#!/bin/bash
cd
export TURTLEBOT3_MODEL=waffle_pi
cd wrsn_rl
source devel/setup.bash
rosrun scheduler dqn_rl_bridge.py
