#!/bin/bash
cd
export TURTLEBOT3_MODEL=waffle_pi
cd tb_simulation
source devel/setup.bash
roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/maps/bcat_testspace.yaml
