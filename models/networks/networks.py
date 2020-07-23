import tensorflow as tf 
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
import numpy as np
class CNN:

    def __init__(self, input_shape, output_shape, activation_function='relu', loss='mse'):
        self.model = models.Sequential()
        self.model.add( layers.Conv2D( 128, kernel_size=(3,3), activation='relu', input_shape=input_shape) )
        self.model.add( layers.MaxPool2D( (2,2) ) )
        self.model.add( layers.Conv2D( 128, kernel_size=(3,3), activation='relu' ) )
        self.model.add( layers.MaxPool2D( (2,2) ) )
        self.model.add( layers.Conv2D( 64, kernel_size=(3,3), activation='relu' ) )
        self.model.add( layers.MaxPool2D( (2,2) ) )


        self.model.add( layers.Flatten() )        
        self.model.add( layers.Dense( output_shape, activation=activation_function ) )

        self.model.compile( optimizer=Adam( learning_rate=1e-3 ), loss=loss,  metrics=["accuracy"])
    

    def train( self, X, Y ):
        self.model.fit( X, Y ,epochs=1 )

    def predict( self, input_image ):
        return self.model.predict(np.array(input_image).reshape((-1, *input_image.shape) ) )[0]

    def save(self, path):
        pass 

    def load(self, path):
        pass 

if __name__ == "__main__":
    cnn = CNN( ( 640, 480, 3 ), 6 )
    print( cnn.model.summary() )