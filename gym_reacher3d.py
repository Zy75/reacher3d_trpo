from roboschool.scene_abstract import SingleRobotEmptyScene
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys

class RoboschoolReacher3d(RoboschoolMujocoXmlEnv):
    def __init__(self):
        RoboschoolMujocoXmlEnv.__init__(self, 'reacher3d.xml', 'body0', action_dim=3, obs_dim=13)

    def create_single_player_scene(self):
        return SingleRobotEmptyScene(gravity=0.0, timestep=0.0165, frame_skip=1)

    TARG_LIMIT = 0.12
    def robot_specific_reset(self):
        self.jdict["target_x"].reset_current_position(self.np_random.uniform( low= 0.0            , high=self.TARG_LIMIT ), 0)
        self.jdict["target_y"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.jdict["target_z"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
 
        self.fingertip = self.parts["fingertip"]
        self.target    = self.parts["target"]
        self.rot_joint = self.jdict["joint0"]
        self.joint_1   = self.jdict["joint1"]
        self.joint_2   = self.jdict["joint2"]
        self.rot_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        self.joint_1.reset_current_position(self.np_random.uniform( low=-3.0, high=3.0 ), 0)
        self.joint_2.reset_current_position(self.np_random.uniform( low=-1.5, high=1.5 ), 0)

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        self.rot_joint.set_motor_torque( 0.05*float(np.clip(a[0], -1, +1)) )
        self.joint_1.set_motor_torque( 0.05*float(np.clip(a[1], -1, +1)) )
        self.joint_2.set_motor_torque( 0.05*float(np.clip(a[2], -1, +1)) )

    def calc_state(self):
        theta,      self.theta_dot = self.rot_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.joint_1.current_relative_position()
        self.beta,  self.beta_dot  = self.joint_2.current_relative_position()
        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        target_z, _ = self.jdict["target_z"].current_position()

#        print target_x,target_y,target_z         

        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
        return np.array([
            target_x,
            target_y,
            target_z,
            self.to_target_vec[0],
            self.to_target_vec[1],
            self.to_target_vec[2],
            np.cos(theta),
            np.sin(theta),
            self.theta_dot,
            self.gamma,
            self.gamma_dot,
            self.beta,
            self.beta_dot,
            ])

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)

    def _step(self, a):
        assert(not self.scene.multiplayer)
        self.apply_action(a)
        self.scene.global_step()

        state = self.calc_state()  # sets self.to_target_vec

        potential_old = self.potential
        self.potential = self.calc_potential()

        electricity_cost = (
            -0.10*(np.abs(a[0]*self.theta_dot) + np.abs(a[1]*self.gamma_dot) + np.abs(a[2]*self.beta_dot))  # work torque*angular_velocity
            -0.01*(np.abs(a[0]) + np.abs(a[1]) + np.abs(a[2]))                                # stall torque require some energy
            )
###        stuck_joint_cost_g = -0.05 if np.abs(np.abs(self.gamma)-1) < 0.01 else 0.0
###        stuck_joint_cost_b = -0.05 if np.abs(np.abs(self.beta )-1) < 0.01 else 0.0
        stuck_joint_cost_g = 0.0
        stuck_joint_cost_b = 0.0
         
        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost_g), float(stuck_joint_cost_b)]
        self.frame  += 1
        self.done   += 0
        self.reward += sum(self.rewards)
        self.HUD(state, a, False)
        return state, sum(self.rewards), False, {}

    def camera_adjust(self):
        x, y, z = self.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)
