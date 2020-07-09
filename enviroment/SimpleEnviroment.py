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

# Hiperparametros
IM_WIDTH = 640 
IM_HEIGHT = 480
SERVER = 'localhost'
PORT = 2000
FOV = 120 #grados

class SimpleEnviroment:



    def __init__(self, model):
        self.client = carla.Client( SERVER, PORT )
        self.client.set_timeout( 5.0 )

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.vehicle_model = self.blueprint_library.filter( model )[0]
        self.front_camera = None

    def get_velocity( self ):
        v = self.vehicle.get_velocity()
        velocity = 3.6 * np.sqrt( v.x ** 2 + v.y ** 2 + v.z ** 2 )
        return velocity


    def image_processing( self, data ):
        img = np.array( data.raw_data )
        img = img.reshape( (IM_HEIGHT, IM_WIDTH, 4) )
        img = img[:, :, :3]
        self.front_camera = img/255.0

    def collision_processing( self, event ):
        self.collisions.append( event )

    def reset( self  ):
        self.collisions = []
        self.actors = []

        spawn_point = np.random.choice( self.world.get_map().get_spawn_points() )
        
        self.vehicle = self.world.spawn_actor( self.vehicle_model, spawn_point )
    
        self.actors.append( self.vehicle )

        #Camera
        camera_blueprint = self.blueprint_library.find( 'sensor.camera.rgb' )
        camera_blueprint.set_attribute( 'image_size_x', f'{IM_WIDTH}' )
        camera_blueprint.set_attribute( 'image_size_y', f'{IM_HEIGHT}' )
        camera_blueprint.set_attribute( 'fov', f'{FOV}' )

        spawn_point = carla.Transform( carla.Location( x=2.5, z=.7 ) ) #camera relative location

        self.camera = self.world.spawn_actor( camera_blueprint, spawn_point, attach_to=self.vehicle )
        self.camera.listen( lambda data: self.image_processing( data ) )


        self.actors.append( self.camera )

        #Colision
        collision_blueprint = self.blueprint_library.find( 'sensor.other.collision' )
        self.collision = self.world.spawn_actor( collision_blueprint, spawn_point, attach_to=self.vehicle )
        self.collision.listen( lambda event: self.collision_processing( event ) )

        self.actors.append( self.collision )
        

        while self.front_camera is None:
            time.sleep( 1e-2 )

        self.start_episode = time.time()
        self.vehicle.apply_control( carla.VehicleControl( throttle=0.0, brake=0.0 ) )

        return self.front_camera

    def step( self, action ):
        reward = 2 
        done = True
        return self.front_camera, reward, done, None