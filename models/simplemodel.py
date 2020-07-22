from  models.networks.Actor import Actor 
from  models.networks.Critic import Critic

class ActorCritic():

    def __init__( self, input_shape, output_shape, min_values, max_values ):
        self.actor = Actor( input_shape, output_shape, min_values, max_values )
        self.critic = Critic( input_shape, output_shape )


    def policy( self, obs ):
        self.mu, self.sigma = self.actor.predict( obs )
        self.value =  self.critic.predict( obs )