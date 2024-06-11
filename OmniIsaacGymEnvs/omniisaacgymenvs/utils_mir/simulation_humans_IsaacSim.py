########################## PYTHON IMPORTS ###############################

import os
import omni
import carb,time
import numpy as np
from omni.isaac.kit import SimulationApp
import asyncio                                 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors    
import multiprocessing          
import time

##########################################################################


########################## ISAAC IMPORTS ###############################

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.world import World
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.kit.commands
from omni.isaac.wheeled_robots.robots import WheeledRobot
from pxr import Gf, Usd, PhysicsSchemaTools, UsdGeom, UsdPhysics       
from omni.isaac.core.utils.stage import is_stage_loading
from omni.physx.scripts import utils
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.controllers import WheelBasePoseController
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.range_sensor import _range_sensor   
from omni.isaac.occupancy_map import _occupancy_map


##########################################################################

############################## CUSTOM IMPORTS ##################################
from Mir100 import MIR100
from randomise_humans import get_human_cmd

##########################################################################


############################SETUP OCCUPANCY MAP############################################################3

def add_toMap(x,y,size,clasif, binary_array = None):
    if binary_array is None:
        binary_array = (image_data > 50).astype(int)
    x_r, y_r = coord_toImg(x,y)
    for x in range(x_r-size, x_r+size):
        for y in range(y_r-size, y_r+size):
            if 0 <= x < binary_array.shape[0] and 0 <= y < binary_array.shape[1]:
                binary_array[x, y] = clasif

    return binary_array

def coord_toImg(x,y):
    x = round(x,1)
    y = round(y,1)
    x_translation = 150
    y_translation = 110
    x_scale = 10
    y_scale = 100 - 110
    
    x_transformed = int(x * x_scale + x_translation)
    y_transformed = int(y * y_scale + y_translation)

    return x_transformed,y_transformed

def plot_occupancy_map():
        
    cmap = colors.ListedColormap(['black', 'white', 'blue', 'red', 'pink'])
    plt.clf()  # Clear the previous plot
    plt.imshow(binary_array, cmap=cmap, origin='lower', norm=colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N))
    plt.colorbar(ticks=[0, 1, 2, 3, 4])
    plt.savefig('plot.png')



############################################## HUMAN TRACKING ##################################################################

def human_toMap(num_char):
    pose_list = []
    for i in range(num_char):
        prims = stage.GetPrimAtPath(f"/World/Characters/Human_n{i}")
        pose_list.append(list(prims.GetAttribute("xformOp:translate").Get()))
        
    
    return pose_list

##############################################################################################################################

##############################################################################################################################

############################## CONFIGURATION ##################################
def main():
    HUMANS = False
    PLOT = True

    ##########################################################################

    if HUMANS:
        num_char = np.random.randint(0,5)
        command = get_human_cmd(num_char)
        text_file = open("Output.txt", "w")
        text_file.write(command)
        text_file.close()

    ########################## SCENE SETUP ###############################

    project_path = os.getcwd()
    warehouse_path = project_path + "/USD/Setup.usd"
    mir_path = project_path + "/USD/MIR.usd"


    wheel_radius = 6.45e-2
    wheel_base = 45e-2

    timeline = omni.timeline.get_timeline_interface()               

    my_world = World(stage_units_in_meters=1, physics_prim_path="/physicsScene")
    stage = omni.usd.get_context().get_stage()



    omni.kit.commands.execute('ToggleExtension',
        ext_id='omni.anim.graph.core-105.1.15',
        enable=True)

    omni.kit.commands.execute('ToggleExtension',
        ext_id='omni.anim.people-0.2.4',
        enable=True)


    omni.kit.commands.execute('ChangeSetting',
        path='/exts/omni.anim.people/navigation_settings/navmesh_enabled',
        value=True)

    omni.kit.commands.execute('ChangeSetting',
        path='/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled',
        value=True)


    omni.kit.commands.execute('SetLightingMenuModeCommand', lighting_mode='stage', usd_context_name='')
    physx_scene = get_prim_at_path("/physicsScene")
    physx_scene.GetAttribute("physxScene:enableGPUDynamics").Set(True)
    add_reference_to_stage(usd_path=warehouse_path, prim_path="/World/warehouse")

    omni.kit.commands.execute('AddGroundPlaneCommand',
        stage=stage,
        planePath='/GroundPlane',
        axis='Z',
        size = 0.0,
        position=Gf.Vec3f(0.0, 0.0, 0.0),
        color=Gf.Vec3f(0.2, 0, 0.5))

    my_mir = my_world.scene.add(
        MIR100(
            prim_path="/World/MIR_ROBOT",
            name="mir100",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            wheel_dof_indices=[1,2],
            create_robot=True,
            usd_path=mir_path,
            position=np.array([np.random.randint(-7,7),np.random.randint(-3,0), 0.1]),
        )
    )

    ##########################################################################


    ########################## HUMAN SETUP ###############################
    if HUMANS:
        from character_setup_widget import CharacterSetupWidget
        wid_human = CharacterSetupWidget()
        wid_human._load_characters_on_clicked(cmd_file_path='Output.txt')
        #wid_human._setup_characters_on_clicked()



    global image_data 
    global binary_array
    image = Image.open('Occupancy_map.png')
    image = image.convert('L')
    image_data = np.array(image)




    ######################## ROBOT CONTROL ################################

    my_mir = my_world.scene.get_object("mir100")


    my_diff_controller = DifferentialController(name="simple_control",wheel_radius=wheel_radius, wheel_base=wheel_base)
    my_controller = WheelBasePoseController(name="pose_control", open_loop_wheel_controller=my_diff_controller,is_holonomic=False)

    my_mir.set_joints_default_state(my_mir.get_joints_state())


    my_diff_controller = DifferentialController(name="simple_control",wheel_radius=wheel_radius, wheel_base=wheel_base)
    my_mir.set_joints_default_state(my_mir.get_joints_state())

    ##########################################################################


    ########################################## SIM RUN ###################################################################
    my_world.reset()



    forward = 0
    rotation = 0 



    while simulation_app.is_running():
        
        my_world.step(render=True)
        timeline.play()                                                
        
        
        #asyncio.run(plot_occupancy_map())
        


        if my_world.is_playing():
            if my_world.current_time_step_index == 0:

                my_diff_controller.reset()
            else:
                mir_position, mir_orientation = my_mir.get_world_pose()
                binary_array = add_toMap(mir_position[0], mir_position[1], 5, 2)
                if HUMANS:
                    list_human = human_toMap(num_char=num_char)
                    for i in list_human:
                        binary_array = add_toMap(i[0],i[1],3,3,binary_array=binary_array)

                wheel_action = my_diff_controller.forward([forward,rotation])
                my_mir.apply_joint_actions(wheel_action)

                if PLOT:
                    if np.random.randint(0,100) in range(5):
                        plot_occupancy_map()

        



                


        #simulation_app.close()

main()
    ##############################################################################################################################

