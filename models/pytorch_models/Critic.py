from models.pytorch_models.network import CNN 


class Critic( CNN ):

    def __init__( self, input_shape, output_shape = 1 ):
        super( Critic, self ).__init__( input_shape, output_shape )

    