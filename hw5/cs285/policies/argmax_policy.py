import numpy as np
import pdb


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def set_critic(self, critic):
        self.critic = critic

    def get_action(self, obs):
        # MJ: changed the dimension check to a 3
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        # TODO: get this from hw3
        actions = self.critic.qa_values(observation)
        # print (actions.shape)
        # raise NameError
        # print (actions.shape)
        action = np.argmax(actions, axis=1)
        # print (actions)

        return action.squeeze()
        
    ####################################
    ####################################