# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from Custom_simulator import get_state ,obj_creation, arranging_sections, passenger_parse, Trains, Junction,find_reward_delay

train_list=[]
section_list = []
junction_list = []
passenger_list = []
line_list = []

asset_list, same_junctions = obj_creation("common_junction.txt", train_list, junction_list, section_list, line_list)
arranging_sections(section_list, line_list)
passenger_data=[]
output_file = "passenger.txt"
for junction in junction_list:
    if junction.signal == 1:
        lambda_value = junction.lambda_value if hasattr(junction, 'lambda_value') else 0.5
        num_passengers = junction.num_passengers if hasattr(junction, 'num_passengers') else 10
        passenger_data = generate_passenger_data(passenger_data, junction, lambda_value, num_passengers, line_list,junction_list, output_file)
write_data_to_file(output_file,passenger_data)
passenger_list = passenger_parse(output_file)

initial_state = get_state(train_list, section_list, junction_list) 
state_size= 


EPISODES = 1000
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # env = gym.make('CartPole-v1')
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = get_state(train_list, section_list, junction_list)
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            # next_state, reward, done, _ = env.step(action)
            reward = find_reward_delay(passenger_list)
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                # Logging training loss every 10 timesteps
                if time % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                        .format(e, EPISODES, time, loss))  
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
        

#states: Train: train speed, train capacity, train no of passengers, next_station, nextstation signal , arriving and leaving time, section
         #Section: train, arriving leaving
         #Junction: signal status, number of passenger
         #passenger: boarding time, leaving time, which train
         
         
#states: Train: train speed, train capacity, train no of passengers, next_station, nextstation signal ,section
         #Junction: signal status, number of passenger
         