#!/bin/bash
gnome-terminal -- bash -c "sleep 10;./navigation_node.sh"
gnome-terminal -- bash -c "sleep 15;./wp_listner.sh"
sleep 15;
./rl_bridge.sh
