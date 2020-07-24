import tensorflow as tf

from models.networks.Actor import Actor 
from models.networks.Critic import Critic
import tensorflow_probability as tfp 
from utils.action_space import Action
import numpy as np
tfd = tfp.distributions

import json


class ActorCritic():

    _MAX_BATCH_SIZE_MEMORY = 128
    _SAFE_DATA = 'safe_data.json'

    def __init__( self, input_shape, output_shape, min_values, max_values, load=False, save_path=None ):
        self.actor = Actor( input_shape, output_shape, min_values, max_values )
        self.critic = Critic( input_shape, 1 )
        self.memory = [] #Aqui pondria mi memoria si la tuviese
        self.action = Action( [], min_values, max_values )
        self.X = []
        self.critic_predictions = []
        self.rewards = []
        self.mean_rewards = []
        self.ephochs = 0

        self.save_path = save_path

        if load and save_path:
            self.load()

    def save( self ):
        self.actor.save( self.save_path )
        self.critic.save( self.save_path )
        safe_data = {
            'mean_rewards': self.mean_rewards,
            'ephochs': self.ephochs
        }
        
        
    def load( self ):
        self.actor.load( self.save_path )
        self.critic.load( self.save_path )
        safe_data = {} #aqui cargaria mi json si tan solo sabria como...

    def policy( self, obs ):
        self.mu, self.sigma = self.actor.predict( obs )
        self.value =  self.critic.predict( obs )
        self.critic_predictions.append( self.value )
        action_distribution = tfd.MultivariateNormalDiag( self.mu, self.sigma )
        return action_distribution

    def get_action( self, obs ):
        self.X.append( obs )
        action_distribution = self.policy( obs )
        action = action_distribution.sample(1)[0]
        action = np.array( action )
        self.action.set_action( action )
        
        return self.action.get_action()

    def learn( self, rewards, final_obs ,done, gamma ):
        td_targets = self.calculate_n_step( rewards, final_obs, done, gamma )
        td_errors = [ td_target - critic_prediction for td_target, critic_prediction in zip( td_targets, self.critic_predictions ) ]
        self.Y_critic = np.array( td_targets )
        self.Y_actor = np.array( td_errors )
        self.X = np.array( self.X )

        self.actor.train( self.X, self.Y_actor )
        self.critic.train( self.X, self.Y_critic )

        self.mean_rewards.append( np.mean( self.rewards ) )

        self.ephochs += 1
        if self.save_path:
            self.save()
        self._reset_variables()

    def calculate_n_step( self, rewards, final_obs, done, gamma ):
        g_t_n_s = []

        g_t_n = np.array( [0.0] ).astype( np.float32 ) if done else self.critic.predict( final_obs )
        for r_t in rewards[::-1]:
            g_t_n = np.array( r_t ).astype( np.float32 ) + gamma * g_t_n 
            g_t_n_s.insert( 0, g_t_n )
        return g_t_n_s
 
    def _reset_variables( self ):
        self.rewards = []
        self.X = []