import argparse
import numpy as np 
import pyvista as pv
from pathlib import Path
from tqdm import tqdm 


parser = argparse.ArgumentParser()
parser.add_argument('ply', help='path to colored pointlcoud object in .ply format')
parser.add_argument('--radius', '-r', help='the radius of the circular path in meters', type=float, default=.1)
parser.add_argument('--distance', '-d', help='the distance z measured from the origin, the circular path is going to take place in meters. this number should be negative for viewing camera reconstructions', type=float, default=-.2)
parser.add_argument('--focus', help='set the viewing direction of the camera, default to the center of the pointcloud', nargs=3, type=float)
parser.add_argument('--period', default=4, type=int)
parser.add_argument('--out','-o', help='path to output video')
parser.add_argument('--resolution', nargs=2, help='output_video_resolution', default=[512,512], type=int)
parser.add_argument('--fps', help='output_video_fps', default=30, type=int)
parser.add_argument('--endoscope', help='path to the endoscope\'s cad model')
parser.add_argument('--grid', help='display grid', action='store_true')
parser.add_argument('--psf', help='pointcloud scale factor', type=float, default=1)
parser.add_argument('--display', help='keep render open', action='store_true')


def render_camera_axis(pv_plotter, scale=0.015):
    x = pv.Arrow(direction=(1,0,0), scale=scale)
    y = pv.Arrow(direction=(0,1,0), scale=scale)
    z = pv.Arrow(direction=(0,0,1), scale=scale)
    
    pv_plotter.add_mesh(x, rgb=True, color='red')
    pv_plotter.add_mesh(y, rgb=True, color='green')
    pv_plotter.add_mesh(z, rgb=True, color='blue')



if __name__ == '__main__':
    args = parser.parse_args()
    
    #configure plotter
    plotter = pv.Plotter(window_size=args.resolution)
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plotter.open_movie(args.out, args.fps)
    plotter.set_background('black')
    if args.grid:
        plotter.show_grid()
    plotter.show_axes()
    plotter.camera_set = True
    render_camera_axis(plotter, 0.015)
    
    
    #add the pointcloud and endoscope
    pt_cloud= pv.read(args.ply)
    pt_cloud.points = pt_cloud.points*args.psf
    plotter.add_mesh(pt_cloud, point_size=1, rgb=True)
    
    if args.endoscope is not None:
        try:
            endoscope = pv.read(args.endoscope)
            plotter.add_mesh(endoscope, rgb=True)
        except FileNotFoundError:
            raise
        
    # define the viewing direction of the camera
    if args.focus is None:
        focus = pt_cloud.center
    else:
        focus = args.focus

    # render all views
    for f in tqdm(range(0,args.fps*args.period),
                  total=args.fps*args.period,
                  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        
        #comptue camera location for frame f
        r = args.radius
        x = r*np.cos((f*2*np.pi/(args.fps*args.period)))
        y = -r*np.sin((f*2*np.pi/(args.fps*args.period)))
        
        # this can potentially be replaced using build-in function 
        plotter.camera.SetPosition(x,y,args.distance) # position the camera to the location computed earlier
        plotter.camera.SetViewUp(0,-1,0) # set the y axis pointing down
        plotter.camera.SetFocalPoint(focus)  # point the z axis towards the center of the object
        plotter.reset_camera_clipping_range() # calculate the clipping plane
        plotter.show(interactive_update=(args.display or (args.out is not None)), auto_close=False) # render view
        if args.out is not None:
            plotter.write_frame()
        
    if args.display:
        print(f'camera position: {plotter.camera_position[0]}')
        print(f'focus point:     {plotter.camera_position[1]}')    
        print(f'up vector:       {plotter.camera_position[2]}')         
    plotter.close()   
