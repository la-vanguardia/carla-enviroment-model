'''
La clase SimplreEnviroment implemente una enviroment madre sencilla 
la cual incluye los metodos mas comunes para lograr implementar un Q-Learner. 


El sensor base elegido sera la camara rgb, debido a su versatilidad



'''

#Imports

import carla 
import numpy as np
import glob
import os
import sys
import cv2
import time


from enviroment.sensors import cameras, collision
from enviroment.rewards import StandardReward

# Hiperparametros
IM_WIDTH = 480 
IM_HEIGHT = 480
SERVER = 'localhost'
PORT = 2000
FOV = 90 #grados
GREEN = 'Green'


class SimpleEnviroment:



    def __init__(self, model):
        self.client = carla.Client( SERVER, PORT )
        self.client.set_timeout( 5.0 )
        self.client.load_world( '/Game/Carla/Maps/Town02' )
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        self.vehicle_model = self.blueprint_library.filter( model )[0]
        

        self.front_camera = None
        self.sensors_spawn_points = carla.Transform( carla.Location( x=2.5, z=.7 ) )
        self.collision_spawn_points = carla.Transform( carla.Location( x=0.0, y=0.0 ) ) 

    
        #handlers
        self.handler_rewards = StandardReward( 10.0 )


    def image_processing( self, data ):
        img = np.array( data.raw_data )
        img = img.reshape( (IM_HEIGHT, IM_WIDTH, 4) )
        img = img[:, :, :3]
        #img = np.reshape(img, (img.shape[2], img.shape[1], img.shape[0]))
        self.front_camera = img / 255

    def collision_processing( self, event ):
        self.collisions.append( event )

    def add_camera(self):
        camera_blueprint = cameras.create_camera_blueprint( IM_WIDTH, IM_HEIGHT, FOV, self.blueprint_library )
        self.camera = self.world.spawn_actor( camera_blueprint, self.sensors_spawn_points, attach_to=self.vehicle )
        self.camera.listen( lambda data: self.image_processing( data ) )
        self.actors.append( self.camera )

    def add_collision(self):
        collision_blueprint = collision.create_collision_blueprint( self.blueprint_library )
        self.collision = self.world.spawn_actor( collision_blueprint, self.collision_spawn_points, attach_to=self.vehicle )
        self.collision.listen( lambda event: self.collision_processing( event ) )
        self.actors.append( self.collision )
        

    def spawn_vehicle( self ):
        spawn_points = self.map.get_spawn_points()
        spawn_point = np.random.choice( spawn_points )
        self.vehicle = self.world.spawn_actor( self.vehicle_model, spawn_point )
        self.actors.append( self.vehicle )

    def destroy_actors( self ):
        self.actors = []

    def reset( self  ):
        self.destroy_actors()
        self.collisions = []

        
        self.spawn_vehicle()
        self.add_camera()
        self.add_collision()


        while self.front_camera is None:
            time.sleep( 1e-2 )

        self.start_episode = time.time()
    

    def step( self, action ):
        self.apply_action( action )
        is_collision = len( self.collisions ) > 0
        reward = self.handler_rewards.compute_reward( self.map, self.vehicle, is_collision)
        done = True if is_collision else False
        return self.front_camera, reward, done, self.start_episode - time.time()

    def apply_action( self, action ):
        control = carla.VehicleControl( throttle=float( action[0] ), steer=float( action[1] ), brake=float( action[2] ))
        self.vehicle.apply_control( control )