import carla
from carla import LaneType, TrafficLightState
import numpy as np


class StandardReward():
    """
        La Clase StandardReward se encarga de computar el premio que recive el agente luego de una acción
    """

    VERY_GOOD = 100
    GOOD = 50
    BAD = -50
    VERY_BAD = -10000

    def __init__(self, junction_threshold):
        self.junction_threshold = junction_threshold
        self.lane_type_detect = LaneType.Driving | LaneType.Sidewalk

    def search_waypoint_type( self, waypoints, lane_type ):
        waypoints_filter = []
        for waypoint in waypoints:
            if waypoint.lane_type == lane_type:
                waypoints_filter.append( waypoint )
        return waypoints_filter

    def compute_reward( self, enviroment_map, vehicle, is_collision ):
        reward = 0
        
        light_state = vehicle.get_traffic_light_state()
        speed_limit = vehicle.get_speed_limit()
        speed = get_speed( vehicle )
        location = vehicle.get_location()

        waypoints_handler = enviroment_map.get_waypoint( location, True, self.lane_type_detect )

        if light_state == TrafficLightState.Red:
            reward = self.red_light_reward( speed, waypoints_handler )
        elif light_state == TrafficLightState.Green:
            reward = self.green_light_reward( speed )
        
        if is_collision:
            reward = self.VERY_BAD


        return reward

    def red_light_reward( self, speed, waypoint_handler ):
        
        waypoints = waypoint_handler.next( self.junction_threshold )
        driving_waypoints = self.search_waypoint_type( waypoints, LaneType.Driving )
        if len( driving_waypoints ) > 0:
            is_junction = driving_waypoints[0].is_junction
        else:
            is_junction = False
        if (is_junction and speed == 0) or ( not is_junction and speed != 0 ):
            reward = self.GOOD
        
        return self.BAD

    def green_light_reward( self, speed ):
        if speed != 0:
            return self.GOOD
        return self.BAD

def get_speed( vehicle ):
        velocity = vehicle.get_velocity()
        speed = 3.6 * np.sqrt( velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2 )
        return speed