from models.networks.networks import CNN

class Critic(CNN):
    
    def __init__(self, input_shape, output_shape):
        super( Critic, self ).__init__( input_shape, output_shape )


    def predict( self, input_image ):
        return self.model.predict( input_image )