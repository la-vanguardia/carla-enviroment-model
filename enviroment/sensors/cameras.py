def create_camera_blueprint( im_width, im_height, fov, blueprints ):
    camera_blueprint = blueprints.find( 'sensor.camera.rgb' )
    camera_blueprint.set_attribute( 'image_size_x', f'{im_width}' )
    camera_blueprint.set_attribute( 'image_size_y', f'{im_height}' )
    camera_blueprint.set_attribute( 'fov', f'{fov}' )

    return camera_blueprint