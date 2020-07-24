class Action:

    def __init__( self, action, min_values, max_values ):
        self.action = action 
        self.min_values = min_values 
        self.max_values = max_values 

    def set_action( self, action ):
        self.action = action

    def get_action( self ):
        self.calmps()
        return self.convert_to_action_space()

    def calmps( self ):
        for i in range( len( self.action ) ):
            self.action[i] = self._calmp( self.action[i], i )

    def _calmp( self, value ,index ):
        if value < self.min_values[ index ]:
            return self.min_values[ index ]
        if value > self.max_values[ index ]:
            return self.max_values[index]
        return value

    def convert_to_action_space( self ):
        throttle = 0.0 
        steer = float( self.action[1] ) 
        brake = 0.0

        if self.action[0] < 0:
            brake = float( -1 * self.action[0] )
        else: 
            throttle = float( self.action[0] )

        return [ throttle, steer, brake ]
        