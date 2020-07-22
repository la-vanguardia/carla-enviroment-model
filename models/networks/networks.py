import tensorflow as tf 
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam

class CNN:

    def __init__(self, input_shape, output_shape, activation_function='relu'):
        self.model = models.Sequential()
        self.model.add( layers.Conv2D( 128, kernel_size=(3,3), activation='relu', input_shape= input_shape ) )
        self.model.add( layers.MaxPool2D( (2,2) ) )
        self.model.add( layers.Conv2D( 128, kernel_size=(3,3), activation='relu' ) )
        self.model.add( layers.MaxPool2D( (2,2) ) )
        self.model.add( layers.Conv2D( 64, kernel_size=(3,3), activation='relu' ) )
        self.model.add( layers.MaxPool2D( (2,2) ) )


        self.model.add( layers.Flatten() )
        self.model.add( layers.Dense( 32, activation='relu' ) )
        self.model.add( layers.Dropout( .2 ) )
        
        self.model.add( layers.Dense( 16, activation='relu' ) )
        self.model.add( layers.Dropout( .2 ) )

        
        self.model.add( layers.Dense( output_shape, activation=activation_function ) )

        self.model.compile( optimizer=Adam( learning_rate=1e-3 ), loss='mse',  metrics=["accuracy"])
    

    def train( self, X, Y ):
        self.model.fit( X, Y, batch_size=64 ,epochs=1 )

    def save(self, path):
        pass 

    def load(self, path):
        pass 

