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


import math
import os
from PIL import Image
import numpy as np
import torch

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.Mir100 import MIR100
from omniisaacgymenvs.robots.articulations.views.MirView import MirView
from omniisaacgymenvs.utils_mir.differential_controller import DifferentialController

from omni.isaac.core.utils.prims import get_prim_at_path

class MirTask(RLTask):
    
#################### ADDED ########################
    def coord_toImg(self, x,y):
        x = round(x,1)
        y = round(y,1)
        x_translation = 150
        y_translation = 110
        x_scale = -10
        y_scale = -10
        

        x_transformed = int(x * x_scale + x_translation)
        y_transformed = int(y * y_scale + y_translation)

        return x_transformed,y_transformed
    
    def add_toMap(self,x,y,size,clasif, binary_array = None):
        if binary_array is None:
            binary_array = (self.image_data > 50).astype(int)
        
        binary_array[self.elevator_corners[1][0]:self.elevator_corners[0][0]:,self.elevator_corners[1][1]:self.elevator_corners[0][1]] = 3
        x_r, y_r = self.coord_toImg(x,y)
        for x in range(x_r-size, x_r+size):
            for y in range(y_r-size, y_r+size):
                if 0 <= x < binary_array.shape[0] and 0 <= y < binary_array.shape[1]:
                    binary_array[x, y] = clasif

        return binary_array
    
    def savePlot(self,i,map_array):
       
        map_array = np.array(map_array)

        # Define color mapping
        colors = [(0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]

        # Create an empty RGB image with the same dimensions as the array
        height, width = map_array.shape
        image = Image.new("RGB", (width, height))

        # Fill in the image pixels based on the array values
        for y in range(height):
            for x in range(width):
                value = map_array[y, x]
                image.putpixel((x, y), colors[value])

        # Save the image as a PNG file
        try:
            image.save(f"Maps/sim{self.simulation}/{i}.png")
        except:
            os.mkdir(os.getcwd()+f"/Maps/sim{self.simulation}")
            image.save(f"Maps/sim{self.simulation}/{i}.png")
            

#######################################################
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._max_episode_length = 12000

        self._num_observations = 47541
        self._num_actions = 2
        
        # Case Specific
        elevator_coords = [[-0.85,6.7],[0.85,9.7]]

        self.elevator_corners = [self.coord_toImg(elevator_coords[0][0],elevator_coords[0][1]),
                                 self.coord_toImg(elevator_coords[1][0],elevator_coords[1][1])]    
        

        #Save files
        self.epoch_rest = 0
        self.simulation = 0

        # Map import
        image = Image.open('utils_mir/Occupancy_map.png')
        image = image.convert('L')
        self.image_data = np.array(image)
        self.my_diff_controller = DifferentialController(name="simple_control",wheel_radius=6.45e-2, wheel_base=45e-2)


        self._mir_position = torch.tensor([0, 0,0])

        RLTask.__init__(self, name=name, env=env)

        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)


        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._dt = self._task_cfg["sim"]["dt"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        

    def set_up_scene(self, scene) -> None:
        self.get_mir()
        super().set_up_scene(scene, replicate_physics='True')
        self._mir = MirView(
            prim_paths_expr="/World/envs/.*/MIR", name="mir_view"
        )
        scene.add(self._mir)
        return
    
    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("mir_view"):
            scene.remove_object("mir_view", registry_only=True)
        self._mir = MirView(
            prim_paths_expr="/World/envs/.*/MIR", name="mir_view"
        )
        scene.add(self._mir)

    def get_mir(self):
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
    
    def get_observations(self) -> dict:
        self.binary_array = [0]*self._mir.count
        self.mir_pos, orientations = self._mir.get_world_poses(clone=False)
        self.mir_pos -= self._env_pos

        for i in range(self._mir.count):
            _mir_pos = self.mir_pos[i].squeeze().tolist()
            self.binary_array[i] = self.add_toMap(_mir_pos[0], _mir_pos[1], 5, 2)
            
            self.obs_buf[i,:] = torch.from_numpy(np.array(self.binary_array[i]).flatten())

            self.epoch_rest +=1

            if self.epoch_rest%(32*40) == 0:
                self.savePlot(self.epoch_rest/(32*40),self.binary_array)
                
                #print(_mir_pos)
                #print('0:',self.dist[0])
        

        observations = {self._mir.name: {"obs_buf": self.obs_buf}}
        return observations
    
    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)
        
        wheel_action = []

        for i in range(self._mir.count):
            wheel_action.append(self.my_diff_controller.forward(actions[i].squeeze().tolist()))

            dof_vel = torch.zeros((self._mir.count,self._mir.num_dof), device=self._device)
            
            dof_vel[:,self._r_dof_idx] = wheel_action[i].joint_velocities[0] 
            dof_vel[:,self._l_dof_idx] = wheel_action[i].joint_velocities[1] 
        

        self._mir.set_joint_velocities(dof_vel)
        
        return


    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        root_pos = self.initial_root_pos.clone()
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0
        self._mir.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._mir.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        root_pos[env_ids, 0] += 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        root_pos[env_ids, 1] += 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        self._mir.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        self._mir.set_velocities(root_velocities[env_ids], indices=env_ids)
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
    def post_reset(self):
        self._l_dof_idx = self._mir.get_dof_index("left_wheel_joint")
        self._r_dof_idx = self._mir.get_dof_index("right_wheel_joint")
        self._mir_dof_idx = self._mir.get_body_index("base_link")   

        self.root_pos, self.root_rot = self._mir.get_world_poses(clone=False)
        self.root_velocities = self._mir.get_velocities(clone=False)
        self.dof_pos = self._mir.get_joint_positions(clone=False)
        self.dof_vel = self._mir.get_joint_velocities(clone=False)
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()
        
        indices = torch.arange(self._mir.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        reward = [0]*self._mir.count
        self.dist = [0]*self._mir.count
        self.state = [0]*self._mir.count
        
        for i in range(self._mir.count):
            if 2 not in self.binary_array[i]:
                reward[i] = -1
            
            else:
                
                x_pos,y_pos = np.where(self.binary_array[i] == 2)
                center = [int((self.elevator_corners[0][0]+self.elevator_corners[1][0])/2),int((self.elevator_corners[0][1]+self.elevator_corners[1][1])/2)]
                dist = math.sqrt((center[0] - x_pos[0]) ** 2 + (center[1] - y_pos[0]) ** 2)
                if self.dist[i] < dist:
                    reward[i] = -np.exp(dist/1000) - self.progress_buf[i].item()*(10**-30)
                    oper = -2
                else:
                    reward[i] = -np.exp(dist/100) - self.progress_buf[i].item()*(10**-30)
                    oper = 2
                
                for x,y in zip(x_pos,y_pos):
                    if x >= 298:
                        x = 297
                        state = -10
                        reward[i] = oper*reward[i]
                        break
                    if y>=158:
                        y=157
                        state = -10
                        reward[i] = oper*reward[i]
                        break

                    _surr = list(set([self.binary_array[i][k, j] for k, j in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]]))

                    if _surr == [2] or _surr==[2,3] or _surr==[3]:
                        state = 10
                        reward[i] = np.exp(dist/100) + self.progress_buf[i].item()*(10**-10)
                        print(f'{i}: GOAL REACHED')
                    elif 4 in _surr:
                        state = 1
                        reward[i] = oper*reward[i]
                        break
                    
                    elif 0 in _surr:
                        state = 1
                        reward[i] = oper*reward[i]
                        break

                    
                    else:
                        state = 0 
                        reward[i] = reward[i]
                        break
                        
                        
                
                    
            
            self.dist[i] = dist
            self.state[i] = state
        self.state = torch.from_numpy(np.array(self.state))
        self.dist = torch.from_numpy(np.array(self.dist))
        self.rew_buf[:] = torch.from_numpy(np.array(reward))
        
        
        




    def is_done(self) -> None:
        
        resets0 = torch.where(self.progress_buf >= self._max_episode_length, 1, 0)
        resets1 = torch.where((self.dist) > 115, 1, 0)
        resets2 = torch.where((self.state) == 1, 1, 0)
        resets3 = torch.where((self.state) == -10, 1, 0)

        resets0 = resets0.to(self._device)
        resets1 = resets1.to(self._device)
        resets2 = resets2.to(self._device)
        resets3 = resets3.to(self._device)

        resets = torch.max(resets0,resets1)
        resets = torch.max(resets,resets2)
        resets = torch.max(resets,resets3)

        
        if resets[0] == 1:
            self.simulation += 1
            self.epoch_rest = 0
            if self.simulation == 20:
                self.simulation = 0
                

        self.reset_buf[:] = resets
        



