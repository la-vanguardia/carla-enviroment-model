from models.networks.networks import CNN
import numpy as np
class Critic(CNN):
    
    def __init__(self, input_shape, output_shape):
        super( Critic, self ).__init__( input_shape, output_shape )
