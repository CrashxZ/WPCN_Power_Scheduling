#!/usr/bin/env python3
import gym 
import threading
import RPi.GPIO as GPIO
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
import os
import csv
import time
from WPCN_ENV import NeedySensor
from datetime import datetime
import rospy
from geometry_msgs.msg import Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from std_msgs.msg import String
import re
from DQNAgent import DQNAgent
from sensor_msgs.msg import BatteryState
from sensor_package.srv import sensor_state

#ros init
rospy.init_node('Reinforcement_Agent_Invoker', anonymous=True)
target = Pose()
target_goal = rospy.Publisher("/waypoint_pose", Pose , queue_size=1)
RELAY_PIN = 17 #pin connected to the amplifier and the power transmitter


#RL Stuff
env=NeedySensor()
csv_log2 = open(env.model_name+".csv", "w")
logger2 = csv.writer(csv_log2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
logger2.writerow(['Episode','Sim-Time','epsilon','total_reward','average_sr', 'Total_Distance'])
state = env.reset()
model_name=NeedySensor.model_name
model_path = NeedySensor.model_path
state_size = env.observation_space.shape
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
batch_size = 128
agent.enc1.trainable = False
agent.enc2.trainable = False
state = env.reset()
battery_rates = []

#working_dir = os.getcwd()
#os.chdir(model_path)
#model.load(model_path+"/"+model_name)
agent.load(env.model_name)
#os.chdir(working_dir)

model_type = re.sub('[^A-Z]', '', str(model))


start_time = datetime.now()
csv_log = open(model_type+"_v_"+NeedySensor.sensor_count+".csv", "w")
logger = csv.writer(csv_log, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
logger2.writerow(['Episode','Sim-Time','epsilon','total_reward','average_sr', 'Total_Distance'])

done = False
action = -1
reward=0
#ros Stuff
def waypoint_coordinator_callback(msg):
    global action
    global state
    global reward
    if msg:
        if msg.data == "0":
            print("Navigating to goal") # ros speed 50 hz
            GPIO.output(RELAY_PIN, GPIO.LOW)
            pass
        elif msg.data == "1":
            state = np.reshape(state, [1, state_size])
            if action != -1 and not done:
                target.position.x = env.sensor_positions[action][0]
                target.position.y = env.sensor_positions[action][1]
                target.orientation.w = env.sensor_positions[action][2]
                target_goal.publish(target)

                env.charge(action) #update the environment
                charge_sensor(action) #physically charge the sensor
                action = agent.act(state)
                env.previous_state = env.state.copy()
                state, reward, done, _ = env.step(action)
                agent.memorize(np.reshape(state, [1, state_size]), action, reward, np.reshape(next_state, [1, state_size]), done)
                state = next_state
                next_state = np.reshape(next_state, [1, state_size])

                if len(agent.memory)>batch_size:
                        agent.replay(batch_size)

                    
                 
                
            else:
                print("Episode Done")
                print("Saving Model")
                agent.memorize(env.previous_state.copy(), action, reward, next_state.copy(), done)
                agent.save(env.model_name)
                rospy.signal_shutdown("Terminal State")


def charge_sensor(action):         
    if not done:
        #call set relay to high
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        time_to_charge = 0
        while sensor_state.read(action) < 100:
            env.background_decay(1,1) #deplete other sensors while charging
            time_to_charge += 1
            time.sleep(1)
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        battery_rates[action].append(state[action][2],env.time_elapsed)
        if battery_rates[action].size > 10:
            b0, b1 = np.polyfit(battery_rates[action][: ,0], battery_rates[action][: ,1], 1)
            env.battery_coeff_B0[action] = b0
            env.battery_coeff_B1[action] = b1

        print("Charged - ", action , " for ", time_to_charge, " seconds")

        

def battery_monitor_callback(self, battery_state):
        time_diff = round(time.time()-self.start_time,1)
        if (time_diff == self.battery_state_count):
            rospy.loginfo("Battery:  Voltage - " + str(battery_state.voltage) + " | Percentage - " +  str(battery_state.percentage * 100))
            self.battery_state_count += 1
            if battery_state.voltage < 10.8 or env.terminal_flag:
                print("Battery Low")
                target.position.x = env.ss[0]
                target.position.y = env.ss[1]
                target.orientation.w = 0
                target_goal.publish(target)
                done = True
                


#function to calculate regression




#ROS callbacks
req_listener_sub = rospy.Subscriber("/request_next", String ,waypoint_coordinator_callback, queue_size=1)
battery_monitor_topic = rospy.Subscriber("/battery_state", BatteryState , battery_monitor_callback, queue_size=1)


while not rospy.is_shutdown():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_PIN, GPIO.OUT)
    #take the first action
    action = agent.act(state)
    time.sleep(0.5)


rospy.spin()










                
