from models.networks.networks import CNN
import numpy as np
from utils.action_space import Action

from os.path import join

class Actor(CNN):

    _SAVE_NAME = 'actor'

    def __init__(self, input_shape, output_shape, min_values, max_values, load=False, path_load=None):
        #Se utilizan el doble de parametros de salida ya que es necesario 
        #tener la media y la varianza de cada accion debido a la politica utilizada
        super( Actor, self ).__init__( input_shape, 2 * output_shape, 'tanh', loss='binary_crossentropy' )

        self.action_mu = Action( [], min_values, max_values )
        self.output_shape = output_shape

        if load and path_load:
            self.load( path_load )


    def predict( self, input_image ):
        prediction = super( Actor, self ).predict( input_image )
        mus, sigmas = prediction[0: self.output_shape ], prediction[self.output_shape:]
        self.action_mu.set_action( mus )
        self.action_mu.calmps()
        sigmas = [ sigma if sigma > 0 else 1e-7 for sigma in sigmas ] 
        return self.action_mu.action, sigmas


    def save( self, path ):
        super( Actor, self ).save( self._get_data_path( path ) )

    def load( self, path ):
        super( Actor, self ).load( self._get_data_path( path ) )

    def _get_data_path( self, path ):
        return join( path, self._SAVE_NAME )



