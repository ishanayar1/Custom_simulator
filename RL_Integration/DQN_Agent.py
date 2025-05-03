import math
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import Input
from keras.optimizers import Adam
from object_creation_modified import find_junction_by_name
import re
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten
from object_creation_modified import find_train_by_name



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                    optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward,next_state):
        self.memory.append((state, action, reward,next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            next_q_values = self.model.predict(next_state, verbose=0)[0]
            target = reward + self.gamma * np.amax(next_q_values)

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def get_state(train_list, junction_list): 
    states = []
    for junction in junction_list:
        state = []

        input_string = junction.name
        number = re.findall(r'\d+', input_string)
        junction_number = int(number[0]) if number else None
        
        train_rem_capacity=0 
        train_number= -1
        if junction.train != "":
            train_string= junction.train
            train_num = re.findall(r'\d+', train_string)
            train_number = int (train_num[0]) 
            train= find_train_by_name (train_list, train_string)     
            train_rem_capacity = train.capacity - train.no_of_passengers
        number_of_passenger_waiting = len(junction.passengerqueue)
        
        state.append(junction_number)
        state.extend([train_number, number_of_passenger_waiting, train_rem_capacity])  
        states.append(state)
        
    number_of_junction = len(junction_list)
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(number_of_junction, 4, 1), padding='same'),
        Flatten(),
        Dense(24, activation='relu'),
    ])
    states_matrix = np.array(states, dtype=np.float32)
    scaler = MinMaxScaler()
    states_matrix = scaler.fit_transform(states_matrix)
    cnn_input = states_matrix.reshape(1, 1, number_of_junction * 4, 1)
    print("cnn_input.shape:", cnn_input.shape)
    flattened_output = model.predict(cnn_input)
    print(f' this is shape {flattened_output.shape}')


    return flattened_output

