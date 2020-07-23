from models.networks.networks import CNN
import numpy as np
from utils.action_space import Action
class Actor(CNN):

    def __init__(self, input_shape, output_shape, min_values, max_values):
        #Se utilizan el doble de parametros de salida ya que es necesario 
        #tener la media y la varianza de cada accion debido a la politica utilizada
        super( Actor, self ).__init__( input_shape, 2 * output_shape, 'tanh', loss='binary_crossentropy' )

        self.action_mu = Action( [], min_values, max_values )
        self.output_shape = output_shape

    def predict( self, input_image ):
        prediction = super( Actor, self ).predict( input_image )
        mus, sigmas = prediction[0: self.output_shape ], prediction[self.output_shape:]
        self.action_mu.set_action( mus )
        sigmas = [ sigma if sigma > 0 else 1e-7 for sigma in sigmas ] 
        return self.action_mu.get_action(), sigmas

