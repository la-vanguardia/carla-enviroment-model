import tensorflow as tf

from models.networks.Actor import Actor 
from models.networks.Critic import Critic
import tensorflow_probability as tfp 

import numpy as np
tfd = tfp.distributions



class ActorCritic():

    def __init__( self, input_shape, output_shape, min_values, max_values ):
        self.actor = Actor( input_shape, output_shape, min_values, max_values )
        self.critic = Critic( input_shape, 1 )
        #refactorizar en 1 clase de tipo Action - para simplicar el codigo!
        self.min_values = min_values
        self.max_values = max_values
        self.X = []
        self.Y_critic = []
        self.Y_actor = []
        self.log_probs = []
        self.critic_predictions = []
        self.rewards = []

   
    def policy( self, obs ):
        self.mu, self.sigma = self.actor.predict( obs )
        self.value =  self.critic.predict( obs )
        self.critic_predictions.append( self.value )
        action_distribution = tfd.MultivariateNormalDiag( self.mu, self.sigma )
        return action_distribution

    def get_action( self, obs ):
        self.X.append( obs )
        action_distribution = self.policy( obs )
        action = action_distribution.sample(1)
        log_prob = action_distribution.log_prob( action )
        action = self.proccess_action( np.array( action[0] ) )
        return action

    def proccess_action( self, action ):
        for i in range( 3 ):
            action[i] = self._calmp( action[i], i )
        return action

    def _calmp( self, value ,index ):
        if value < self.min_values[ index ]:
            return self.min_values[ index ]
        if value > self.max_values[ index ]:
            return self.max_values[index]
        return value

    def learn( self, rewards, final_obs ,done, gamma ):
        td_targets = self.calculate_n_step( rewards, final_obs, done, gamma )
        td_errors = [ td_target - critic_prediction for td_target, critic_prediction in zip( td_targets, self.critic_predictions ) ]
        self.Y_critic = np.array( td_targets )
        self.Y_actor = np.array( td_errors )
        self.X = np.array( self.X )
        self.actor.train( self.X, self.Y_actor )
        self.critic.train( self.X, self.Y_critic )
        self.X = []

    def calculate_n_step( self, rewards, final_obs, done, gamma ):
        g_t_n_s = []

        g_t_n = np.array( [0.0] ).astype( np.float32 ) if done else self.critic.predict( final_obs )
        for r_t in rewards[::-1]:
            g_t_n = np.array( r_t ).astype( np.float32 ) + gamma * g_t_n 
            g_t_n_s.insert( 0, g_t_n )
        return g_t_n_s
 