import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import concurrent.futures
import os

from utils import Portfolio


# references:
# https://arxiv.org/pdf/1802.09477.pdf
# https://arxiv.org/pdf/1509.06461.pdf
# https://papers.nips.cc/paper/3964-double-q-learning.pdf
class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.model_type = 'DQN'
        self.state_dim = state_dim
        self.action_dim = 5  # hold, buy, sell, pending_buy, pending_sell
        self.memory = deque(maxlen=2000)
        self.buffer_size = 2880 

        self.tau = 0.0001
        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.990 # decrease exploration rate as the agent becomes good at trading
        self.is_eval = is_eval

        self.model = load_model(f'saved_models/{model_name}.h5') if is_eval else self.model()
        self.model_target = load_model(f'saved_models/{model_name}_target.h5') if is_eval else self.model
        self.model_target.set_weights(self.model.get_weights()) # hard copy model parameters to target model parameters

        if not self.is_eval:
            self.tensorboard = TensorBoard(log_dir='./logs/DDQN_tensorboard', update_freq='epoch')
            self.tensorboard.set_model(self.model)

    def update_model_target(self):
        model_weights = self.model.get_weights()
        model_target_weights = self.model_target.get_weights()
        for i in range(len(model_weights)):
            model_target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * model_target_weights[i]
        self.model_target.set_weights(model_target_weights)

    def model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    @tf.function
    def silent_predict(model, state):
        return model(state, training=False)

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        options = Agent.silent_predict(self.model, state)
        return np.argmax(options.numpy()[0])

    def experience_replay(self):
        mini_batch = random.sample(self.memory, self.buffer_size)

        def predict_next_state(next_state):
            return self.model_target.predict(next_state, verbose=0)[0]

        def predict_state(state):
            return self.model.predict(state, verbose=0)

        with concurrent.futures.ThreadPoolExecutor(max_workers = min(32, (os.cpu_count() or 1) + 4)) as executor:
            next_states = [tup[3] for tup in mini_batch]
            states = [tup[0] for tup in mini_batch]

            future_next_states = {executor.submit(predict_next_state, next_state): next_state for next_state in next_states}
            future_states = {executor.submit(predict_state, state): state for state in states}

            Q_expected_values = []
            next_actions_list = []

            for future in concurrent.futures.as_completed(future_next_states):
                next_state = future_next_states[future]
                Q_expected = future.result()
                Q_expected_values.append(Q_expected)

            for future in concurrent.futures.as_completed(future_states):
                state = future_states[future]
                next_actions = future.result()
                next_actions_list.append(next_actions)

        for i, (state, actions, reward, next_state, done) in enumerate(mini_batch):
            Q_expected = reward + (1 - done) * self.gamma * np.amax(Q_expected_values[i])
            next_actions_list[i][0][np.argmax(actions)] = Q_expected
            history = self.model.fit(state, next_actions_list[i], epochs=1, verbose=0, callbacks=[])

        self.update_model_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]