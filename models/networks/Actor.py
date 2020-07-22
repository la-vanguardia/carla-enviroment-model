from models.networks.networks import CNN

class Actor(CNN):

    def __init__(self, input_shape, output_shape, min_values, max_values):
        #Se utilizan el doble de parametros de salida ya que es necesario 
        #tener la media y la varianza de cada accion debido a la politica utilizada
        super( Actor, self ).__init__( input_shape, 2 * output_shape, 'softplus' )
        self.min_values = min_values
        self.max_values = max_values
        self.output_shape = output_shape

    def predict( self, input_image ):
        prediction = self.model.predict( input_image )
        mu, sigma = prediction[0: self.output_shape + 1 ], prediction[self.output_shape:]
        for i in range( self.output_shape ):
            mu[i] = self._calmp( mu[i], i )
            sigma[i] = sigma[i] if sigma[i] > 0 else 1e-7
        return mu, sigma
    
    def _calmp( self, value ,index ):
        if value < self.min_values[ index ]:
            return self.min_values[ index ]
        if value > self.max_values[ index ]:
            return self.max_values
        return value