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

minutes = 5

t = np.linspace( 0, 60 * minutes, 20 * 60 * minutes )

for i in range( 20 * 60 * minutes ):
    data, reward, _, semaforo = enviroment.step('')
    out.write( data )
    rewards.append( reward )
    rewards_data.write( semaforo + '\n')
    time.sleep( 1/20 )

out.release()
rewards_data.close()

plt.plot( t, rewards )
plt.show()
plt.savefig( './rewads.png' )
