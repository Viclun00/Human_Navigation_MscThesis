# Human_Navigation_MscThesis
Human-aware Mobile Robot Navigation in Simulated Production Environments Master Thesis. By Victor Luna Santos


For running the training and the model, follow the instructions of OmniIsaacGymEnvs found in https://github.com/isaac-sim/OmniIsaacGymEnvs, install and run with task name task=Mir

In the file OmniIsaacGymEnvs/omniisaacgymenvs/task/MIR100.py you can change human scenario, what is logged, reward function etc.  In the file OmniIsaacGymEnvs/omniisaacgymenvs/cfg/train/MirPPO.py you can change the training hyperparameters. For more information, follow https://github.com/isaac-sim/OmniIsaacGymEnvs

In the Camnera directory there is the script for real time human input in the map, people_distance_Realtime_segment.py It is just plug&play.

In the DataMgmt directory, you can find the data treatment notebooks to check your own trainings and inference. 
