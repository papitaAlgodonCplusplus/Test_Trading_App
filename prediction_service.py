import numpy as np

def predict_action(agent, state):
    actions = agent.model.predict(state, verbose=0)[0]
    action = agent.act(state)
    return action, actions