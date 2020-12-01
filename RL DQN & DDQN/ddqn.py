import gym
import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pickle
import os.path
from os import path
from IPython.display import display, clear_output
from gym import wrappers

class Memory:
  def __init__(self, dim):
    self.current = 0
    self.has = 0
    self.batchsize = 32
    self.maxsize = 1000000

    self.states = np.empty((self.maxsize, dim))
    self.new_states = np.empty((self.maxsize, dim))
    self.rewards = np.empty(self.maxsize)
    self.actions = np.empty(self.maxsize, dtype=int)
    self.dones = np.empty(self.maxsize)
  
  def remember(self, state, action, reward, new_state, done):
    self.states[self.current] = state
    self.new_states[self.current] = new_state
    self.rewards[self.current] = reward
    self.actions[self.current] = action
    self.dones[self.current] = 1 if done else 0

    self.has = min(self.has+1, self.maxsize)
    self.current = (self.current+1)%self.maxsize
  
  def canLearn (self):
  	if self.has < self.batchsize:
  		return False
  	return True

  def samples(self):
    if self.has < self.batchsize:
      return [[], [], [], [], []]
    
    indexes = np.random.randint(0, self.has, self.batchsize)
    return [self.states[indexes], self.new_states[indexes], self.rewards[indexes], self.actions[indexes], self.dones[indexes]]

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = Memory(self.env.observation_space.shape[0])
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay_duration = 1e6
        self.learning_rate = 0.0000625
        
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(128, input_dim=state_shape[0], activation="relu", kernel_initializer=keras.initializers.VarianceScaling(scale=2), bias_initializer=keras.initializers.Zeros()))
        model.add(Dense(128, activation="relu", kernel_initializer=keras.initializers.VarianceScaling(scale=2), bias_initializer=keras.initializers.Zeros()))
        model.add(Dense(128, activation="relu", kernel_initializer=keras.initializers.VarianceScaling(scale=2), bias_initializer=keras.initializers.Zeros()))
        model.add(Dense(self.env.action_space.n, kernel_initializer=keras.initializers.VarianceScaling(scale=2), bias_initializer=keras.initializers.Zeros()))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.remember(state, action, reward, new_state, done)

    def replay(self):
        states, new_states, rewards, actions, dones = self.memory.samples()

        if len(states) == 0:
          return
        
        arg_q_max = np.argmax(self.model.predict(new_states), axis=1)
        q_vals = self.model.predict(states)
        Q_futures = self.target_model.predict(new_states)[np.arange(0,32), arg_q_max]
        q_vals[np.arange(0,32), actions] = rewards + Q_futures*self.gamma*(1-dones)
        loss_hist = self.model.fit(states, q_vals, epochs=1, verbose=0)

        return [loss_hist.history['loss'][0], np.mean(Q_futures), np.mean(q_vals)]
    
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def epsilonDecay (self, frames):
        t = min(frames/self.epsilon_decay_duration, 1)
        self.epsilon = self.epsilon_start*(1-t) + self.epsilon_min*t

    def act(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])
    
    def save_model(self, fn):
        self.model.save(fn)

def evaluate(dqn_agent, frames):
	evprogress = {'rewards': [], 'frames': []}
	if os.path.exists('eval.pickle'):
		with open('eval.pickle', 'rb') as handle:
			evprogress = pickle.load(handle)
	dqn_agent.epsilon = 0.05

	evid = len(evprogress['frames'])

	env = gym.make("SpaceInvaders-ram-v0")
	env = wrappers.Monitor(env, "vids/"+str(evid), force=True)

	trials  = 1
	trial_len = 10000
	total_reward = 0

	for trial in range(trials):
		cur_state = env.reset()
		cur_state = cur_state[np.newaxis,:]
		trial_reward = 0
		for step in range(trial_len):
			action = dqn_agent.act(cur_state)
			new_state, reward, done, _ = env.step(action) 

			new_state = new_state[np.newaxis,:]
			reward = clip_reward(reward)

			trial_reward += reward

			cur_state = new_state
			if done:
				break
		total_reward += trial_reward
	print("evaluated",evid)
	print("reward",total_reward/trials)
	print("")
	evprogress['frames'].append(frames)
	evprogress['rewards'].append(total_reward/trials)
	with open('eval.pickle', 'wb') as handle:
		pickle.dump(evprogress, handle)

	dqn_agent.epsilonDecay(frames)

def main():
    env = gym.make("Breakout-ram-v0")
    trials  = 50000
    trial_len = 10000
    train_target_length = 10000
    UPDATE_FREQ = 4
    frames = 0
    maxreward = 0
    total_reward = 0
    dqn_agent = DQN(env=env)
    if os.path.exists('dqn.model'):
      dqn_agent.model.load_weights('dqn.model')
      dqn_agent.target_model.load_weights('dqn.model')
    progress = {'time': [], 'rewards': [], 'frames': [], 'epsilon': [], 'losses': [], 'Qs': [], 'maxQs': [], 'total_frames': 0}
    if os.path.exists('dqn.pickle'):
      with open('dqn.pickle', 'rb') as handle:
        progress = pickle.load(handle)
        frames = progress['total_frames']
        dqn_agent.epsilonDecay(frames)
    for trial in range(trials):
        cur_state = env.reset()
        cur_state = cur_state[np.newaxis,:]
        trial_reward = 0
        start = timer()
        trial_frames = 0

        weight_updates = 0
        total_losses = 0
        total_maxQs = 0
        total_Qs = 0

        for step in range(trial_len):
            frames += 1
            trial_frames += 1

            action = dqn_agent.act(cur_state)
            #env.render()
            new_state, reward, done, _ = env.step(action) 

            new_state = new_state[np.newaxis,:]
            reward = clip_reward(reward)

            trial_reward += reward
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            if frames%UPDATE_FREQ == 0 and dqn_agent.memory.canLearn():
              loss, maxQs, Qs = dqn_agent.replay()
              weight_updates += 1
              total_losses += loss
              total_maxQs += maxQs
              total_Qs += Qs
            if frames%train_target_length == 0:  
              dqn_agent.target_train()
            dqn_agent.epsilonDecay(frames)
            cur_state = new_state
            if done:
                break
        end = timer()

        maxreward = max(maxreward, trial_reward)
        total_reward += trial_reward
        print("ran",trial)
        print("trial time",end - start)
        print("reward",trial_reward)
        print("epsilon",dqn_agent.epsilon)
        print("frames",frames)
        print("max reward",maxreward)
        print("avg reward",total_reward/(trial+1))
        progress['time'].append(end - start)
        progress['rewards'].append(trial_reward)
        progress['frames'].append(trial_frames)
        progress['total_frames'] = frames
        progress['epsilon'].append(dqn_agent.epsilon)

        progress['losses'].append(total_losses/weight_updates)
        progress['maxQs'].append(total_maxQs/weight_updates)
        progress['Qs'].append(total_Qs/weight_updates)

        with open('dqn.pickle', 'wb') as handle:
          pickle.dump(progress, handle)
        dqn_agent.save_model("dqn.model")

        if trial%1000 == 0:
        	evaluate(dqn_agent, frames)
    return

def clip_reward(reward):
  if reward > 0:
    return 1
  elif reward == 0:
   return 0
  else:
    return -1
main()