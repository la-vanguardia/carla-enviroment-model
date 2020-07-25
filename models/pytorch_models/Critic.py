from models.pytorch_models.network import CNN 


class Critic( CNN ):

    def __init__( self, input_shape, output_shape = 1, load=False, load_path=None ):
        super( Critic, self ).__init__( input_shape, output_shape, load, load_path )

    