import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maximizes the Q-value 
        # at the current observation as the output
        actions = self.critic.qa_values(observation)
        # print (actions.shape)
        # raise NameError
        # print (actions.shape)
        action = np.argmax(actions, axis=1)
        # print (actions)

        return action.squeeze()