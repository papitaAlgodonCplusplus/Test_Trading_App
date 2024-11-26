import sys 
import os 
import random 
from collections import deque 
from contextlib import redirect_stdout 
import numpy as np 
import tensorflow as tf 
from keras.layers import Dense 
from keras.models import load_model 
from keras.optimizers import Adam 
import datetime
import json
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow logging
tf.get_logger().setLevel('ERROR')  # Disable TensorFlow logging
tf.autograph.set_verbosity(0)

from utils import Portfolio

class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name="DDQN"):
        super().__init__(balance=balance)
        self.model_type = 'DDQN'
        self.model_name = model_name
        self.model_dir = 'saved_models'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy_instant, sell_instant
        self.memory = deque(maxlen=20000)  # Larger buffer for more diverse experience
        self.buffer_size = 5120  # Smaller batches for more stable training

        self.tau = 0.01  # Faster soft update for dynamic environments
        self.gamma = 0.95  # Long-term reward focus
        self.epsilon = 0.95  # Initial exploration rate
        self.epsilon_min = 0.05  # Minimum exploration rate
        self.epsilon_decay = 0.95  # Slower decay for prolonged exploration
        self.is_eval = is_eval
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())  # Sync weights

    def update_target_model(self):
        """Soft update the target model parameters."""
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

    def create_model(self):
        """Define the neural network architecture with separate outputs for action and quantity."""
        input_layer = tf.keras.Input(shape=(self.state_dim,))

        # Shared layers
        dense_1 = Dense(128, activation='relu')(input_layer)
        dense_2 = Dense(64, activation='relu')(dense_1)
        dense_3 = Dense(32, activation='relu')(dense_2)

        # Action output
        action_output = Dense(self.action_dim, activation='linear', name='action_output')(dense_3)

        # Quantity output
        quantity_output = Dense(1, activation='relu', name='quantity_output')(dense_3)  # Positive quantities

        model = tf.keras.Model(inputs=input_layer, outputs=[action_output, quantity_output])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'action_output': 'huber_loss', 'quantity_output': 'mse'},  # Separate losses for each output
            loss_weights={'action_output': 1.0, 'quantity_output': 0.5},  # Weight action loss higher
            run_eagerly=False
        )
        return model

    def reset(self):
        """Reset the agent's state."""
        self.reset_portfolio()
        # self.epsilon *= 1.5  # Increase exploration rate for new episodes
        print("Resetting agent state...", self.balance, self.inventory)

    def remember(self, state, actions, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, actions, reward, next_state, done))

    @staticmethod
    @tf.function
    def silent_predict(model, state):
        """Silent prediction without logging."""
        return model(state, training=False)

    def act(self, state, max_quantity=50, forbidden_actions=[]):
        """
        Select an action and predict the quantity based on the current policy.

        Args:
            state: Current state of the environment.
            max_quantity: Maximum quantity allowed for buy/sell actions.

        Returns:
            tuple: (action, quantity)
        """
        if not self.is_eval and np.random.rand() <= self.epsilon:
            # Random exploration for action and quantity
            action = random.randrange(self.action_dim)
            quantity = random.uniform(1, max_quantity)  # Random quantity for exploration
            return action, quantity

        # Predict action and quantity
        action_predictions, quantity_predictions = self.model.predict(state, verbose=0)

        # Get the chosen action
        action = np.argmax(action_predictions[0])

        # Get the predicted quantity (ensure it's within limits)
        quantity = np.clip(quantity_predictions[0][0], 1, max_quantity)

        return action, quantity

    def experience_replay(self):
        """Train the model using experience replay."""
        if len(self.memory) < self.buffer_size:
            return  # Skip if not enough memory

        mini_batch = random.sample(self.memory, self.buffer_size)

        # Ensure correct input shapes
        next_states = np.array([np.squeeze(tup[3]) for tup in mini_batch])  # Remove extra dimensions
        states = np.array([np.squeeze(tup[0]) for tup in mini_batch])

        actions = [tup[1] for tup in mini_batch]
        rewards = [tup[2] for tup in mini_batch]
        dones = [tup[4] for tup in mini_batch]

        # Predict Q-values for current and next states (extract action predictions)
        Qnext = self.target_model.predict(next_states, verbose=0)  # Only use action predictions
        Qcurrent = self.model.predict(states, verbose=0)  # Only use action predictions

        # Validate shapes
        assert Qnext.shape[0] == len(mini_batch), "Mismatch between Q_next and mini_batch size"
        assert Qcurrent.shape[0] == len(mini_batch), "Mismatch between Q_current and mini_batch size"

        # Update Q-values
        for i in range(len(mini_batch)):
            Q_target = rewards[i]
            if not dones[i]:  # If not terminal, add future discounted reward
                Q_target += self.gamma * np.amax(Qnext[i])
            Qcurrent[i][np.argmax(actions[i])] = Q_target

        # Train the model
        history = self.model.fit(states, Qcurrent, batch_size=32, verbose=0, callbacks=[])

        # Update the target model
        self.update_target_model()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history.get('loss', [0])[0]

    def save_models(self, episode=None):
            """
            Save models and training history with versioning.
            
            Args:
                episode: Current episode number for versioning
            """
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"_ep{episode}" if episode is not None else ""
            
            # Save models
            from pathlib import Path

            # Assuming self.model_dir is a string, convert it to a Path object
            model_dir_path = Path(self.model_dir)

            # Save the model using the Path object
            self.model.save(model_dir_path / f"{self.model_name}{version}_{timestamp}.h5")
            self.target_model.save(model_dir_path / f"{self.model_name}{version}_{timestamp}_target.h5")
            
            print(f"Models and history saved with version: {version}_{timestamp}")

    def load_models(self):
        """Load the latest version of models and training history."""
        try:
            # Find latest model files
            model_files = list(self.model_dir.glob(f"{self.model_name}*.h5"))
            latest_model = max(
                (f for f in model_files if not f.stem.endswith('_target')),
                key=lambda x: x.stat().st_mtime
            )
            latest_target = max(
                (f for f in model_files if f.stem.endswith('_target')),
                key=lambda x: x.stat().st_mtime
            )
            
            # Load models
            self.model = load_model(latest_model)
            self.target_model = load_model(latest_target)
            
            print(f"Models loaded successfully from: {latest_model.name}")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise