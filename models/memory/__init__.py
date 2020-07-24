import numpy as np 

class Memory():

    def __init__( self, max_batch_size ):
        self._memory = []
        self._max_batch_size = max_batch_size

    
    def add( self, memory ):
        if len( self._memory ) >= self._max_batch_size:
            del self._memory[0]
        
        self._memory.append( memory )

    def random_memory( self ):
        return np.random.choice( self._memory, 1 )