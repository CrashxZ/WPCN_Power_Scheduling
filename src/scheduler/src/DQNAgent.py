# # -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.optimizers import Adam , SGD 
import matplotlib.pyplot as plt

EPISODES = 5000
log_path = "/home/mainak/wrsn_navigation_rl/rl_agents_cleaned/logs/"
callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq =1)



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.sensor_count = 10
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = 2048
        self.memory = deque(maxlen=self.memory_size)
        self.gamma = 0.8# discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.991
        self.learning_rate =  0.0001
        #self.model_class = nn_model(batch_size,self.sensor_count,3)
        self.temp_action = -1
        self.path_arr = []
        self.schedular = -1
        self.loss_list = []
        self.behaviour_probs = np.zeros(action_size)
        self.target_probs = np.zeros(action_size)
        self.total_steps = 0
        self.r_count = 0
        self.action_count = np.zeros(self.sensor_count)
        #self.target_model = self._build_model()
        #self.target_train()
        self.train_count = 0
        self.d = 0.3
        
        #NN Layers
        # Encoder
        self.enc1 = Dense(512, activation="relu")
        self.enc2 = Dense(512, activation="relu")
        self.final_enc = Dense(self.action_size, activation="sigmoid")
        # NN Layers
        self.mlp1 = Dense(512, activation='relu')
        self.mlp2 = Dense(512, activation='relu')
        self.mlp3 = Dense(512, activation='relu')
        self.out = Dense(self.action_size, activation='relu')
        self.final = Dense(self.action_size, activation='linear')
        
        self.model = self._build_model()
        
        


    def _build_model(self):
        state_input = Input(self.state_size)
        mlp = self.enc1(state_input)
        mlp = self.enc2(mlp)
        mlp = Flatten()(mlp)
        mlp = self.final_enc(mlp)
        mlp = self.mlp1(mlp)
        mlp = self.mlp2(mlp)
        mlp = self.mlp3(mlp)
        mlp = self.out(mlp)
        final = self.final(mlp)
        model = tf.keras.Model(state_input, final)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
        model.summary()
        return model
        


        
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        
    def visualize(self,state,action,reward,next_state,q_value):
        print(action, state[0][action],state[0][self.sensor_count+action] ,next_state[0][action],next_state[0][self.sensor_count+action])
        for i in range(self.sensor_count):
            if i == action:
                print("{} || {} || {} || {} || {} || {} <---".format(state[0][i], state[0][self.sensor_count+i], next_state[0][i], next_state[0][self.sensor_count+i], reward , q_value[0][i]))
            else:
                print("{} || {} || {} || {} || {} || {}".format(state[0][i], state[0][self.sensor_count+i], next_state[0][i], next_state[0][self.sensor_count+i], reward, q_value[0][i]))
        print("------------------------------------------------------------", state[0][self.sensor_count+i+2:])
        print("------------------------------------------------------------", next_state[0][self.sensor_count+i+2:])
    
    def visualize_action(self,state,action,q_value):
        for i in range(self.sensor_count):
            if i == action:
                print("{} || {} || {} || {} <---".format(state[i][0], state[i][1], state[i][2], q_value[0][i]))
            else:
                print("{} || {} || {} || {}".format(state[i][0], state[i][1], state[i][2], q_value[0][i]))
        print("------------------------------------------------------------")

    #temporary backup schedular
    def backup_schedular(self,state):
        x = [item[1] for item in state]
        y = [item[0] for item in state]
        y_indices = np.argsort(y)
        x_indices = np.argsort(x)
        if len(self.path_arr) > self.sensor_count-2:
            self.path_arr = self.path_arr[:self.sensor_count-5]
        for index in x_indices:
            if index in self.path_arr:
                continue
            else:
                action = index
                self.path_arr = np.insert(self.path_arr,0,action)
                self.behaviour_probs[action] += 1
                return action
                    
    def act(self, state,robot_speed):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            self.schedular = 0
            self.temp_action = self.backup_schedular(state)
            return self.temp_action
        self.schedular = 1
        q_values = self.model.predict(state.reshape(1,self.sensor_count,3))
        action = np.argmax(q_values)
        self.path_arr = np.insert(self.path_arr,0,action)
        self.target_probs[action] += 1
        # Fallback
        if(state[action][1]/robot_speed > state[action][0]/ state[action][0]):
            action = self.backup_schedular(state)
        return action

    def supervised_learn(self,state,action,reward,done):
        q_values = np.zeros(self.sensor_count)
        max_dist = np.amax([item[1] for item in state])
        min_dist = np.amin([item[1] for item in state])
        r = max_dist - min_dist
        for i in range(self.sensor_count):
            q_values[i]=(1-state[i][0]/100) * 0.5 #- (state[i][1]/r)
        
        q_values[action] += reward
        return q_values.reshape(1,self.sensor_count).copy()
            
    
    def train(self):
        minibatch = self.memory
        states = []
        q_list = []
        for state, action, reward, next_state, done in minibatch:
            q_values = self.supervised_learn(state,action,reward,done)
            state = state.reshape(1,self.sensor_count,3).copy()
            states.append([state.copy()])
            q_list.append([q_values.copy()])
        states = np.array(states).reshape(self.memory_size,self.sensor_count,3)
        q_list = np.array(q_list).reshape(self.memory_size,self.sensor_count)
        history = self.model.fit(states, q_list, epochs=1, verbose=0)
        self.train_count += 1
        
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        q_list = []
        for state, action, reward, next_state, done in minibatch:
            self.action_count[action] += 1
            target = reward
            if not done:
                # if self.epsilon > 0.5:
                #     val = self.supervised_learn(state,action,reward,done)
                # else:
                val = self.model.predict(next_state.reshape(1,self.sensor_count,3))   
                target = reward + (self.gamma * np.amax(val))
                    
            state = state.reshape(1,self.sensor_count,3).copy()
            q_values = self.model.predict(state)
            #print(q_values.shape)
            q_values[0][action] = target
            states.append([state.copy()])
            q_list.append([q_values.copy()])
        
        states = np.array(states).reshape(batch_size,self.sensor_count,3)
        q_list = np.array(q_list).reshape(batch_size,self.sensor_count)
        
        print(states.shape)
        print(q_list.shape)
        
        history = self.model.fit(states, q_list, epochs=1, verbose=0)
        max_action = np.argmax(self.action_count)
        min_action = np.argmin(self.action_count)
        print("Action --> ",max_action , "Frequency - ",self.action_count[max_action], " || Action --> ",min_action , "Frequency - ",self.action_count[min_action] )
        self.loss_list.append(history.history['loss'][0])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def target_train(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)
        
        
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


        
                
