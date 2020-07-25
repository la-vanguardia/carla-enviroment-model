from models.pytorch_models.network import CNN
from torch import nn
class Actor( CNN ):

    def __init__( self, input_shape, output_shape ):
        super( Actor, self ).__init__( input_shape, 8 )
        self.mu = nn.Sequential(
            nn.Linear( 8, output_shape )
        )
        self.sigma = nn.Linear( 8, output_shape )

    def forward( self, x ):
        x = super( Actor, self ).forward( x )
        return self.mu( x ), self.sigma( x )
