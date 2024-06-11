from typing import Optional
import numpy as np
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.types import XFormPrimState, ArticulationAction


class MirView(ArticulationView):
    def __init__(self, prim_paths_expr: str, name: Optional[str] = "MirView") -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        '''self.mir = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/MIR_ROBOT", name="mir_view", reset_xform_properties=False
        )'''

    
    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view=physics_sim_view)
        
        self._wheel_dof_indices = [self.get_dof_index("left_wheel_joint"), self.get_dof_index("right_wheel_joint")]
        self._num_wheel_dof = len(self._wheel_dof_indices)

        return
    
    @property
    def wheel_dof_indices(self):
        return self._wheel_dof_indices
    


