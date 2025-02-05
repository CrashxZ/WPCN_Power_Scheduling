#!/bin/bash
cd
export TURTLEBOT3_MODEL=waffle_pi
cd wrsn_navigation_rl
source devel/setup.bash
rosrun rl_sch wp_follow.py
