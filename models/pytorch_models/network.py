import torch 

from torch import nn
import numpy as np

class CNN( nn.Module ):

    def __init__( self, input_shape, output_shape ):
        super( CNN, self ).__init__()

        device_name = 'cpu'
        self.device = torch.device( device_name )

        self.layer_conv = nn.Sequential( 
            nn.Conv2d( input_shape[0], 128, 3, stride=3, padding=0 ),
            nn.ReLU(),
            nn.MaxPool2d( 2 ),
            nn.Conv2d( 128, 128, 3, stride=2, padding=0 ),
            nn.ReLU(),
            nn.MaxPool2d( 2 ),
            nn.Conv2d( 128, 64, 3, stride=1, padding=0 ),
            nn.ReLU(),
            nn.MaxPool2d( 2 ),
            nn.Flatten(),
            nn.Linear( 6144 ,output_shape )

        )


    def forward( self, x ):
        x.to( self.device )
        x = self.layer_conv( x )
        return x

    def save( self ):
        return self.state_dict()

    def load( self, state_dict ):
        self.load_state_dict( state_dict )


cnn = CNN( (3, 480, 640), 6 )