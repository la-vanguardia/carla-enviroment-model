import torch 

from torch import nn
import numpy as np

class CNN( nn.Module ):

    def __init__( self, input_shape, output_shape, load=False, load_path=None ):
        super( CNN, self ).__init__()

        device_name = 'cuda' if torch.cuda.is_available else 'cpu'
        self.device = torch.device( device_name )

        print( f'Se entrenara utilizando { self.device }' )

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

    def save( self, path ):
        pass 

    def load( self, path ):
        pass


cnn = CNN( (3, 480, 640), 6 )