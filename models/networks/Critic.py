from models.networks.networks import CNN
import numpy as np
from os.path import join

class Critic(CNN):
    
    _SAVE_NAME = 'critic.ckpt'

    def __init__(self, input_shape, output_shape, load=False, path_load=None):
        super( Critic, self ).__init__( input_shape, output_shape )
        if load and path_load:
            self.load( path_load )

    def save( self, path ):
        super( Critic, self ).save( self._get_data_path( path ) )

    def load( self, path ):
        super( Critic, self ).load( self._get_data_path( path ) )
    
    def _get_data_path( self, path ):
        return join( path, self._SAVE_NAME )

