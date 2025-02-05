# import gym 
import threading
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
import os
from stable_baselines3 import PPO, DQN , A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime
import math
from scipy.spatial import distance
import time
import csv


class NeedySensor(Env):
    model_name = "DQN10"
    model_path = "Training/Models/"+model_name
    log_path = "Training/Logs/"+model_name
    sens_pos = []
    def __init__(self):
        self.sensor_count = 10
        self.critical_level = 10 #percent charge
        self.battery_coeff_B0 = np.full(self.sensor_count, 0.16)
        self.battery_coeff_B1 = np.full(self.sensor_count, 0.14)
        self.pose_x = 0
        self.pose_y = 0
        self.current_position=(self.pose_x,self.pose_y)
        self.sensor_positions = []
        self.ss = [0,0]
        self.total_distance = 0
        self.dead_sensors = 0
        self.min_dist = -1
        self.min_batt = -1
        self.navigation_targets = []
        self.previous_state = []
        self.path_trav = []
        self.survival_rates = []
        self.time_elapsed = 0
        self.time_elapsed_t = 0
        self.time_penalty = 0
        self.terminal_flag = False

        ## Battery info
        ## Min = 54.7 Max = 100 (scaled from 0 to 1)

        with open('/home/mainak/wrsn_navigation_rl/rl_agents_cleaned/waypoints10.txt', mode='r') as waypoints_file:
            waypoints = csv.reader(waypoints_file, delimiter=',')
            for row in waypoints:
                if(row[3]=="1"):
                    self.sensor_positions.append([float(row[0]),float(row[1])])
                    #print('Loading trajectory waypoint - > {}'.format(row))
        print("Loaded sensor positions")
        self.path_trav = [-1,-1,-1,-1,-1]
        self.init_time = datetime.now()
        
        self.action_space = Discrete(self.sensor_count)
        self.observation_space = Box(0,100,shape=(self.sensor_count,3))
        self.state = np.empty([self.sensor_count,3])
        self.P_list = np.zeros(self.sensor_count)
        self.time_between_steps =  0



    # e_i, d_i, r_i

    
    def get_survival_rate(self):
        return (self.sensor_count-self.get_critical_sensor_count())/self.sensor_count

    def add_nav_targets(self,x,y,w,s):
        self.navigation_targets.append([x,y,w,s])

    def distance(self,a,b):
        return round(distance.euclidean(a, b),1)
    
    def update_distances(self):
        for i in range(self.sensor_count):
            #print(i , self.sensor_positions[i])
            self.state[i][1] = self.distance(self.current_position,self.sensor_positions[i])
    
    def update_rates(self):
        t = self.time_elapsed - self.time_elapsed_t 
        for i in range(self.sensor_count):
            self.state[i][2] = (self.state[i][0] - self.previous_state[i][0])/t
        self.time_elapsed_t  = self.time_elapsed
        self.time_penalty = t/2000
    

    def get_min_distance_arr(self):
        x = [item[1] for item in self.state]
        x_indices = np.argsort(x)
        return x_indices
    
    def get_critical_sensor_count(self):
        count = 0
        for i in range(self.sensor_count):
            if self.state[i][0] < self.critical_level:
                count += 1
        return count
    
    def background_decay(self,distance,speed):
        for i in np.arange(-1,distance,speed):
            self.time_elapsed +=1
            #deplete all wireless devices
            for i in range(self.sensor_count):
                val = round(self.state[i][0] - (random.random()*self.battery_coeff_B0[i] + random.random()*self.battery_coeff_B1[i]),2)
                if val > 0:
                    self.state[i][0] = val
                else:
                    self.state[i][0] = 0
    

    def step(self, action):
        done = False
        self.update_rates()
        sr = self.get_survival_rate()
        dist_arr = self.get_min_distance_arr()
        dist_arr_pos = np.where(dist_arr == action)[0][0]
        sd = (dist_arr_pos/self.sensor_count)
        max_dist = np.amax([item[1] for item in state])
        min_dist = np.amin([item[1] for item in state])
        r = max_dist - min_dist
        #no repetation  
        if action in self.path_trav:
            reward = -5
            print(sd,dist_arr_pos, sr , action)
            #done = True
        else:
            reward = 0
            self.path_trav.append(action)
            
            reward = (1-self.state[action][0]/self.hyperparameter) - (self.state[action][1]/r)
            
            
                    
        if sr<0.6:
            done = True
            reward = -10
        
        if len(self.path_trav) > 5:
            self.path_trav = self.path_trav[1:].copy()

        self.P_list = np.insert(self.P_list,0,action)
        self.P_list = self.P_list[:-1]
        self.survival_rates.append(sr)
        distance_travelled = self.state[action][1]
        self.total_distance += distance_travelled
        try:
            info = {'Action':action,'Distance_Travelled':distance_travelled,'Total_Distance':self.total_distance,'Dead_Sensors':self.dead_sensors,'Reward':reward}
        except:
            print(action,self.state)

        return self.state, round(reward, 5), done, info

    def charge(self,sensor_index):
        #deplete the battery levels whicle charging : 50 - current power level / 2 
        charging_time_required = 50 - (self.state[sensor_index][0] / 2)
        self.background_decay(charging_time_required,1)
        self.state[sensor_index][0] = 100
        self.state[sensor_index][2] = self.previous_state[sensor_index][2]
        self.current_position = self.sensor_positions[sensor_index]
        self.pose_x=self.sensor_positions[sensor_index][0]
        self.pose_y=self.sensor_positions[sensor_index][1]

    def reset(self):
        for i in range(self.sensor_count):
            self.state[i][0] = random.randint(40,80)
            self.state[i][2] = -0.1
        self.path_trav = [-1,-1,-1,-1,-1]
        self.total_distance = 0
        self.current_position = (0,0)
        self.pose_x=0
        self.pose_y=0
        self.update_distances()
        self.survival_rates = [1]
        self.previous_state = []
        self.time_elapsed = 0
        self.time_elapsed_t = 0
        return self.state