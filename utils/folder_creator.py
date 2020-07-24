import os 

def is_exist( folder_path ):
    return os.path.isdir( folder_path )

def create_folder( folder, path ):
    full_path = os.path.join( path, folder )
    if not is_exist( full_path ):
        os.mkdir( full_path )

def recursive_folder( folder ):
    folders = folder.split( '/' )
    path = './'
    for folder in folders:
        create_folder( folder, path )
        path = os.path.join( path, folder )

