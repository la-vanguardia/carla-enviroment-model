import glob
import os
import sys

import time

import matplotlib.pyplot as plt
import numpy as np

from enviroment.SimpleEnviroment import SimpleEnviroment 

import cv2

enviroment = SimpleEnviroment('model3')

enviroment.reset()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

rewards = []
rewards_data = open( './data.txt', 'w' )
enviroment.vehicle.set_autopilot( True )
semaforos = []

minutes = 1

t = np.linspace( 0, 60 * minutes, 20 * 60 * minutes )

map_run = enviroment.world.get_map()

print( map_run.get_all_landmarks() )

for i in range( 20 * 60 * minutes ):
    data, reward, _, info = enviroment.step('')
    position = ( 30,30 )
    data = cv2.resize( data, (640, 480))
    cv2.putText(
     data, #numpy array on which text is written
     f"{ enviroment.vehicle.get_speed_limit() } - { info }", #text
     position, #position at which writing has to start
     cv2.FONT_HERSHEY_SIMPLEX, #font family
     1,
     (255, 0, 0) ) 
    out.write( data )
    rewards.append( reward )
 
    time.sleep( 1/20 )


out.release()
rewards_data.close()


