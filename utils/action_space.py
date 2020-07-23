class Action:

    def __init__( self, action, min_values, max_values ):
        self.action = action 
        self.min_values = min_values 
        self.max_values = max_values 

    def set_action( self, action ):
        self.action = action

    def get_action( self ):
        self.calmps()
        return self.action

    def calmps( self ):
        for i in range( len( self.action ) ):
            self.action[i] = self._calmp( self.action[i], i )

    def _calmp( self, value ,index ):
        if value < self.min_values[ index ]:
            return self.min_values[ index ]
        if value > self.max_values[ index ]:
            return self.max_values[index]
        return value