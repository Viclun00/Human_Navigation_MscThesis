# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cv2
import os
from PIL import Image
import numpy as np
import torch
from gym import spaces

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.Mir100 import MIR100
from omniisaacgymenvs.robots.articulations.views.MirView import MirView
from omniisaacgymenvs.utils_mir.differential_controller import DifferentialController
from omniisaacgymenvs.utils_mir.human_sim_grids import get_hum_cases,get_random_paths

from omni.isaac.core.utils.prims import get_prim_at_path


class MirTask(RLTask):
    
#################### ADDED FUNCTIONS########################

    def coord_toImg(self, x,y):
        #Transforms the metric coordinates to image coordinates
        x = round(x,1)
        y = round(y,1)
        x_translation = 5
        y_translation = 12
        x_scale = 2
        y_scale = 2
         

        x_transformed = int(x * x_scale + x_translation)
        y_transformed = int(y * y_scale + y_translation)

        return x_transformed,y_transformed
    
    def add_toMap(self,x,y,size,clasif, binary_array = None, metric_coords=True):
        #Add new object to the map
        if binary_array is None:
            binary_array = (self.image_data > 50).astype(int)
            binary_array[self.elevator_corners[1][0]:self.elevator_corners[0][0]:,self.elevator_corners[1][1]:self.elevator_corners[0][1]] = 3

        if metric_coords:   
            x_r, y_r = self.coord_toImg(x,y)
        else:
            x_r, y_r = x,y
        if size > 0:
            for x in range(x_r-size, x_r+size):
                for y in range(y_r-size, y_r+size):
                    if 0 <= x < binary_array.shape[0] and 0 <= y < binary_array.shape[1]:
                        binary_array[x, y] = clasif
        else:
            x = x_r
            y = y_r
            if 0 <= x < binary_array.shape[0] and 0 <= y < binary_array.shape[1]:
                        binary_array[x, y] = clasif
       
            
        return binary_array
    
    def savePlot(self,j,a=True):
        # Calculate the dimensions of the layout
        num_rows = 1
        num_cols = 1

        # Calculate the dimensions of each binary array
        array_height, array_width = self.binary_array.shape
        print(self.binary_array)   
        # Calculate the dimensions of the combined image
        combined_height = array_height * num_rows
        combined_width = array_width * num_cols

        # Create an empty combined array
        combined_array = np.zeros((combined_height, combined_width), dtype=np.uint8)

        colors = [(0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
        combined_image = Image.fromarray(combined_array).convert('RGB')

        height, width = combined_array.shape

        for y in range(height):
            for x in range(width):
                value = self.binary_array[y, x]
                combined_image.putpixel((x, y), colors[value])

        # Display the image on screen
        opencv_image = np.array(combined_image)
        opencv_image = opencv_image[:, :, ::-1].copy()
        cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image Window', 600, 600)
        cv2.imshow('Image Window', opencv_image)


        cv2.namedWindow('Reward Window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Reward Window', 400, 100)  
        reward_cum = self.reward_cum.cpu().numpy() if self.reward_cum.is_cuda else self.reward_cum.numpy()
        image_height = 100
        image_width = 400
        background_color = (255, 255, 255)  # white background
        rimage = np.full((image_height, image_width, 3), background_color, np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 0)  # black color
        line_type = 2
        start_x = 30
        start_y = 30
        line_spacing = 30
        for i, reward in enumerate(reward_cum):
            position = (start_x, start_y + i * line_spacing)
            text = f"Env {i}: {reward}"
            cv2.putText(rimage, text, position, font, font_scale, font_color, line_type)
        cv2.imshow("Reward Window", rimage)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows() 


        # Save the image as a PNG file in Maps
        if a:
            try:
                combined_image.save(f"Maps/sim{self.simulation}/{j}.png")
            except:
                os.mkdir(os.getcwd()+f"/Maps/sim{self.simulation}")
                combined_image.save(f"Maps/sim{self.simulation}/{j}.png")
        else:
            combined_image.save(f"Maps/{j,self.reward[0][0]}.png")
        return

            

############################TASK  FUNCTIONS###########################
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._max_episode_length = 1000

        
        # Case Specific Scenario
        
        self.width = 24
        self.height = 25
        self._num_observations = self.width * self.height * 3
        self._num_actions = 2
        

        self.observation_space = spaces.Box(
            np.ones((self.width, self.height, 3), dtype=np.float32) * -np.Inf, 
            np.ones((self.width, self.height, 3), dtype=np.float32) * np.Inf)

        self.elevator_coords = [[10,1],[6.8,-1]]

        self.elevator_corners = [self.coord_toImg(self.elevator_coords[0][0],self.elevator_coords[0][1]),
                                 self.coord_toImg(self.elevator_coords[1][0],self.elevator_coords[1][1])]    
        
        #Save files
        self.epoch_rest = 0
        self.simulation = 0

        # Map import - case specific
        image = Image.open('utils_mir/Small_map.png')
        image = image.convert('L')
        self.image_data = np.array(image)
        self.my_diff_controller = DifferentialController(name="simple_control",wheel_radius=6.45e-2, wheel_base=45e-2)
            
        self.clean_binary_array = (self.image_data > 50).astype(int)
        self.clean_binary_array[self.elevator_corners[1][0]:self.elevator_corners[0][0]:,self.elevator_corners[1][1]:self.elevator_corners[0][1]] = 3

        self._mir_position = torch.tensor([0, 0,0.01])

        RLTask.__init__(self, name=name, env=env)

        #Initialize variables for later data mgmt
        self.human_poses = [0]*self._num_envs
        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
        self.pre_dist = torch.ones(self._num_envs, device=self._device) * 10
        self.human_cases = get_hum_cases()
        self.human_state = torch.ones((self._num_envs), device=self._device)*-1


        self.a = 3  #SCENARIO TO SIMULATE FIRST
    
        self.reward_cum = torch.zeros((self._num_envs), device=self._device)

        self.success = 0
        self.time = 0

        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._dt = self._task_cfg["sim"]["dt"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        

    def set_up_scene(self, scene) -> None: #Import the robot in the environment
        self.get_mir()
        super().set_up_scene(scene, replicate_physics='True')
        self._mir = MirView(
            prim_paths_expr="/World/envs/.*/MIR", name="mir_view"
        )
        scene.add(self._mir)


        return
    
    def initialize_views(self, scene): #Start the robot view
        super().initialize_views(scene)
        if scene.object_exists("mir_view"):
            scene.remove_object("mir_view", registry_only=True)
        self._mir = MirView(
            prim_paths_expr="/World/envs/.*/MIR", name="mir_view"
        )
        scene.add(self._mir)

    def get_mir(self): #Create a robot variable
        mir= MIR100(
            prim_path=self.default_zero_env_path + "/MIR",
            name="Mir",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            wheel_dof_indices=[4,5],
            create_robot=True,
            usd_path=os.getcwd()+"/USD/MIR2.usd",
            
        )
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "Mir", get_prim_at_path(mir.prim_path), self._sim_config.parse_actor_config("Mir")
        )
    def cleanup(self) -> None:
        # initialize remaining buffers
        RLTask.cleanup(self)

        # override observation buffer for camera data
        self.obs_buf = torch.zeros(
            (self.num_envs, self.width, self.height,3), device=self.device, dtype=torch.float)

    def get_observations(self) -> dict:  #Get the observations from the environment

        self.human_state = torch.where((self.progress_buf % 64 == 0)|(self.progress_buf==0), self.human_state+1, self.human_state)
        human_state = self.human_state.tolist()
        human_coords = self.human_poses
        human_arrays = []
        binary_arrays = [0]*len(human_state)
       

        for i in range(len(human_state)): #Simulate humans in map
            H_ARRAY = self.clean_binary_array.copy()
            
            if self.a == 1:
                human_arrays.append(H_ARRAY)
            else:
                try:
                    for keys in human_coords[i].keys():
                        try:
                            H_ARRAY = np.array(self.add_toMap(human_coords[i][keys][int(human_state[i])][0], human_coords[i][keys][int(human_state[i])][1], 0, 4, H_ARRAY,metric_coords=False))
                        except:
                            H_ARRAY = np.array(self.add_toMap(human_coords[i][keys][-1][0], human_coords[i][keys][-1][1], 0, 4, H_ARRAY,metric_coords=False))
                    human_arrays.append(H_ARRAY)

                except:
                    human_arrays.append(H_ARRAY)

            


        self.mir_pos, orientations = self._mir.get_world_poses(clone=False)

        root_positions = self.mir_pos - self._env_pos

        _mir_pos = root_positions.squeeze().tolist()
        _mir_orientations = orientations.squeeze().tolist()
        #Add robot to the map
        try:
            binary_arrays = np.array([self.add_toMap(_pos[0], _pos[1], 0, 2, human_arrays[0]) for _pos in _mir_pos])
        except:
            binary_arrays = np.array(self.add_toMap(_mir_pos[0], _mir_pos[1], 0, 2,human_arrays[0]))

        
        colors = [(0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
        rgb_array = np.zeros((binary_arrays.shape[0],binary_arrays.shape[1],3), dtype=np.uint8)
        for i, color in enumerate(colors):
            rgb_array[binary_arrays == i] = color
        import torchvision.transforms as transforms
        image = Image.fromarray(rgb_array, 'RGB')
        transform = transforms.Compose([
        transforms.ToTensor()  # Convert PIL Image to PyTorch tensor
        ])

        # Apply the transformation to the image
        tensor = transform(image)
        tensor = tensor.unsqueeze(0)
        self.obs_buf = torch.swapaxes(tensor, 1, 3).clone().float()
        self.epoch_rest = self.epoch_rest + 1

        observations = {self._mir.name: {"obs_buf": self.obs_buf}}
        
        return self.obs_buf
    
    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.clone().to(self._device)
        self.actions = actions
        
        wheel_action = []

        for i in range(self._mir.count):
            wheel_action.append(self.my_diff_controller.forward(actions[i].squeeze().tolist()))

            self.dof_vel = torch.zeros((self._mir.count,self._mir.num_dof), device=self._device)
            
            self.dof_vel[:,self._r_dof_idx] = wheel_action[i].joint_velocities[0] 
            self.dof_vel[:,self._l_dof_idx] = wheel_action[i].joint_velocities[1] 
        
        
        self._mir.set_joint_velocities(self.dof_vel)

        
        return


    def reset_idx(self, env_ids): #Reset the environment
        num_resets = len(env_ids)

        self.dof_vel = self._mir.get_joint_velocities()
        

        with open(f"time_log_{self.a}_transfer.txt", "a") as log_file: #log inference
            log_file.write(f"Success:{self.success};Time:{self.time};Rew:{int(self.reward_cum)}\n") #Only self.reward_cum for training rewards log

        
        self.reward_cum[env_ids] = 0
        root_pos = self.initial_root_pos.clone()
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0
        self._mir.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._mir.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)

        root_pos[env_ids, 0] += 0.1 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        root_pos[env_ids, 1] += 0.1 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        self._mir.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        self._mir.set_velocities(root_velocities[env_ids], indices=env_ids)
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.pre_dist[env_ids] = 8.2    

        

        for i in range(len(env_ids)):
            self.a = 3 #Select human scene after reset
            if self.a != 1:
                self.human_poses[env_ids[i]]= get_random_paths(get_hum_cases(),self.a,self.clean_binary_array)
                self.human_state[env_ids[i]] = 0
        
        self.success = 0
        self.time = 0



    def post_reset(self): #Reset the robot views
        self._l_dof_idx = self._mir.get_dof_index("left_wheel_joint")
        self._r_dof_idx = self._mir.get_dof_index("right_wheel_joint")
        self._mir_dof_idx = self._mir.get_body_index("base_link")   

        self.root_pos, self.root_rot = self._mir.get_world_poses(clone=False)
        self.root_velocities = self._mir.get_velocities(clone=False)
        self.dof_pos = self._mir.get_joint_positions(clone=False)
        self.dof_vel = self._mir.get_joint_velocities(clone=False)
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {
            "rew_pos": torch_zeros(),
            "raw_dist": torch_zeros()
            
        }
        
        self.target_positions = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self.actions = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)


        self.indices = torch.arange(self._mir.count, dtype=torch.int64, device=self._device)
        self.reset_idx(self.indices)

    def calculate_metrics(self) -> None: #Calculate the reward
        human_state = self.human_state.tolist()
        human_coords = self.human_poses

        human_arrays = []
        binary_arrays = [0]*len(human_state)
        for i in range(len(human_state)):
            H_ARRAY = self.clean_binary_array.copy()
            if self.a == 1:
                human_arrays.append(H_ARRAY)
            else:
                try:
                    for keys in human_coords[i].keys():
                        try:
                            H_ARRAY = np.array(self.add_toMap(human_coords[i][keys][int(human_state[i])][0], human_coords[i][keys][int(human_state[i])][1], 0, 4, H_ARRAY,metric_coords=False))
                        except:
                            H_ARRAY = np.array(self.add_toMap(human_coords[i][keys][-1][0], human_coords[i][keys][-1][1], 0, 4, H_ARRAY,metric_coords=False))
                        
                    human_arrays.append(H_ARRAY)    

                except:
                    human_arrays.append(H_ARRAY)    
                

        self.mir_pos_, orientations = self._mir.get_world_poses(clone=False)
        root_positions = self.mir_pos - self._env_pos
        _mir_pos = root_positions.squeeze().tolist()
        torch_ones = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        center = [int((self.elevator_coords[0][0]+self.elevator_coords[1][0])/2),int((self.elevator_coords[0][1]+self.elevator_coords[1][1])/2)]

        num_sets = len(self.indices)
        envs_long = self.indices.long()
        self.target_positions[envs_long, 0] = torch.ones(num_sets, device=self._device) * center[0]
        self.target_positions[envs_long, 1] = torch.ones(num_sets, device=self._device) * center[1] 

        self.target_dist = torch.sqrt(torch.square(self.target_positions - root_positions[:,:2]).sum(-1))
        pos_reward = self.pre_dist - self.target_dist       

        self.pre_dist = torch.where(self.target_dist < self.pre_dist, self.target_dist, self.pre_dist)

        binary_arrays = np.array(self.add_toMap(_mir_pos[0], _mir_pos[1], 0, 2, human_arrays[0]))
        self.binary_array = binary_arrays


        if self.epoch_rest%(32) == 0:
            self.savePlot(self.epoch_rest/32)
        
    
        binary_array = binary_arrays
        height, width = binary_array.shape


        x_pos,y_pos = np.where(binary_array == 2)
        state = 1
        try:
            _surr = [
                    int(binary_array[min(height-1, x_pos + 1), y_pos]),               # Right
                    int(binary_array[max(0, x_pos - 1), y_pos]),                      # Left
                    int(binary_array[x_pos, min(width-1, y_pos + 1)]),               # Down
                    int(binary_array[x_pos, max(0, y_pos - 1)]),                     # Up
                    int(binary_array[min(height-1, x_pos + 1), min(width-1, y_pos + 1)]),   # Bottom-right
                    int(binary_array[min(height-1, x_pos + 1), max(0, y_pos - 1)]),  # Top-right
                    int(binary_array[max(0, x_pos - 1), min(width-1, y_pos + 1)]),   # Bottom-left
                    int(binary_array[max(0, x_pos - 1), max(0, y_pos - 1)])          # Top-left
]
                            
            if i == 0: 
                print(_surr)

            if _surr.count(0)>3:
                state = -0.1

            elif 4 in _surr: 
                state = -10

            elif (3 in _surr) and (1 not in _surr) and (0 not in _surr):
                state = 30
                print(state)

            elif (0 in _surr) and (3 not in _surr):
                state = -0.01
            
            elif 3 in _surr:
                state = 5

            

            else:
                state = 1
            
            if self.target_dist[i] > 9:
                state = -0.1

            if _surr.count(3) > 4 and self.success == 0:
                self.time = self.epoch_rest
                self.success = 1

        except:
            state = -0.1

        torch_ones = torch_ones*state
        

        #Base reward inverse to distance depending on the state
        pos_reward = torch.where(torch_ones > 5 , 100*1/(1 + self.target_dist), 1/(1 + self.target_dist)) 
        pos_reward = torch.where(torch_ones == 5 , 10*1/(1 + self.target_dist), pos_reward)
        pos_reward = torch.where((torch_ones == 1)&(self.target_dist<3) , 4/(1 + self.target_dist), pos_reward)
        pos_reward = torch.where((torch_ones == 1)&(self.target_dist>self.pre_dist) , -1/(1 + self.target_dist), pos_reward)
        pos_reward = torch.where(torch_ones < 0 , -self.target_dist, pos_reward)
        pos_reward = torch.where((torch_ones >=5)&(self.target_dist>self.pre_dist) , -10/(1 + self.target_dist), pos_reward)

        




        self.episode_sums["rew_pos"] += pos_reward
        self.episode_sums["raw_dist"] += self.target_dist
        self.reward = [pos_reward,torch_ones]

        self.state = torch_ones
        self.rew_buf[:] = pos_reward
        

        self.episode_sums["rew_pos"] += pos_reward
        self.episode_sums["raw_dist"] += self.target_dist
        self.reward = [pos_reward,torch_ones]

        self.reward_cum += pos_reward

        self.state = torch_ones
        self.rew_buf[:] = pos_reward

        
        
    def is_done(self) -> None: #Reset conditions
        
        
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.state < 1, ones, die)
        #die = torch.where(self.state == 10, ones, die)
        check = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)


        if check[0] == 1: #Count simulation steps and resets
            self.simulation += 1
            self.epoch_rest = 0
            if self.simulation == 20:
                self.simulation = 0
                

        



