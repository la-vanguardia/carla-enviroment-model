import os 
import platform
import json

def read_system():
    system = platform.system()
    return system.lower()

def system_configuration():
    system = read_system()
    path = os.path.join( os.getcwd(), f'utils/configs/{ system }.json' )
    with open( path, 'r' ) as json_file:
        configuration = json.load( json_file )
    return configuration


