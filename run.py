import glob
import os
import sys

import matplotlib.pyplot as plt

from enviroment.SimpleEnviroment import SimpleEnviroment 


enviroment = SimpleEnviroment('model3')

enviroment.reset()
img, _, _,_ = enviroment.step( 0 )

plt.imshow( img )
plt.show()