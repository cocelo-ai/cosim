from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco
from envs.flamingo_v1_3.manager.control_manager import ControlManager
from envs.flamingo_v1_3.manager.xml_manager import XMLManager
from envs.flamingo_v1_3.utils.math_utils import MathUtils
from envs.flamingo_v1_3.utils.mujoco_utils import MuJoCoUtils
from envs.flamingo_v1_3.utils.noise_generator_utils import truncated_gaussian_noisy_data, uniform_noisy_data
import glfw


class FlamingoV1_3(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    def __init__(self, config, render_flag=True, render_mode='human'):
        # Set Basic Properties
        self.id = "flamingo_v1_3"
        self.config = config
        self.state_dim = 20
        self.action_dim = 8
        self.command_dim = config["env"]["command_dim"]
        self.render_mode = render_mode
        self.render_flag = render_flag

        # PD control parameters
        self.kp_hip = config["hardware"]["Kp_hip"]
        self.kp_shoulder = config["hardware"]["Kp_shoulder"]
        self.kp_leg = config["hardware"]["Kp_leg"]
        self.kp_wheel = config["hardware"]["Kp_wheel"]

        self.kd_hip = config["hardware"]["Kd_hip"]
        self.kd_shoulder = config["hardware"]["Kd_shoulder"]
        self.kd_leg = config["hardware"]["Kd_leg"]
        self.kd_wheel = config["hardware"]["Kd_wheel"]

        self.action_scaler = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 20.0, 20.0]


        # Set Simulation Properties
        precision_level = self.config["random"]["precision"]
        sensor_noise_level = self.config["random"]["sensor_noise"]
        self.init_noise = self.config["random"]["init_noise"]
        self.dt_ = config["random_table"]["precision"][precision_level]["timestep"]
        self.frame_skip = config["random_table"]["precision"][precision_level]["frame_skip"]
        self.sensor_noise_map = config["random_table"]["sensor_noise"][sensor_noise_level]

        # Set Placeholders
        self.action = np.zeros(self.action_dim)
        self.filtered_action = np.zeros(self.action_dim)
        self.previous_action = np.zeros(self.action_dim)
        self.computed_torques = np.zeros(self.action_dim)
        self.applied_torques = np.zeros(self.action_dim)
        self.obs = None
        self.viewer = None
        self.mode = None

        # Set XML (MJCF)
        self.xml_manager = XMLManager(config)
        self.model_path = self.xml_manager.get_model_path()
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(
            self,
            model_path=self.model_path,
            frame_skip=self.frame_skip,
            observation_space=Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32,),
            render_mode=self.render_mode if render_flag else None,
        )

        # Set other Managers and Helpers
        self.control_manager = ControlManager(config)
        self.mujoco_utils = MuJoCoUtils(self.model)

        # Set Indices of q and qd
        qpos_joint_names = ["left_hip_joint", "right_hip_joint", "left_shoulder_joint", "right_shoulder_joint", "left_leg_joint", "right_leg_joint"]
        qvel_joint_names = ["left_hip_joint", "right_hip_joint", "left_shoulder_joint", "right_shoulder_joint", "left_leg_joint", "right_leg_joint", "left_wheel_joint", "right_wheel_joint"]
        self.q_indices = self.mujoco_utils.get_qpos_joint_indices_by_name(qpos_joint_names)
        self.qd_indices = self.mujoco_utils.get_qvel_joint_indices_by_name(qvel_joint_names)

    def _get_obs(self):
        q = self.data.qpos[self.q_indices]
        qd = self.data.qvel[self.qd_indices]
        quat = self.data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        if np.all(quat == 0):
            quat = np.array([0, 0, 0, 1])

        omega = self.data.sensor('angular-velocity').data.astype(np.double)
        euler = MathUtils.quaternion_to_euler_array(quat)

        q = truncated_gaussian_noisy_data(q, mean=self.sensor_noise_map["q"]["mean"], std=self.sensor_noise_map["q"]["std"],
                                          lower=self.sensor_noise_map["q"]["lower"], upper=self.sensor_noise_map["q"]["upper"])
        qd = truncated_gaussian_noisy_data(qd, mean=self.sensor_noise_map["qd"]["mean"], std=self.sensor_noise_map["qd"]["std"],
                                          lower=self.sensor_noise_map["qd"]["lower"], upper=self.sensor_noise_map["qd"]["upper"])
        omega = truncated_gaussian_noisy_data(omega, mean=self.sensor_noise_map["omega"]["mean"], std=self.sensor_noise_map["omega"]["std"],
                                          lower=self.sensor_noise_map["omega"]["lower"], upper=self.sensor_noise_map["omega"]["upper"])
        euler = truncated_gaussian_noisy_data(euler, mean=self.sensor_noise_map["euler"]["mean"], std=self.sensor_noise_map["euler"]["std"],
                                          lower=self.sensor_noise_map["euler"]["lower"], upper=self.sensor_noise_map["euler"]["upper"])
        obs = np.concatenate([q, qd, omega, euler])
        return obs

    def step(self, action):
        self.action = action
        self.filtered_action = self.control_manager.delay_filter(action)

        # Pull the current joint positions and velocities
        q = self.data.qpos[self.q_indices]
        qd = self.data.qvel[self.qd_indices]

        # Extract joint positions and velocities from observation
        pos_hip = q[0:2]
        pos_shoulder = q[2:4]
        pos_leg = q[4:6]
        pos_wheel = self.data.qpos[[10, 14]]

        vel_hip = qd[0:2]
        vel_shoulder = qd[2:4]
        vel_leg = qd[4:6]
        vel_wheel = qd[6:8]

        hip_action_scaled = self.filtered_action[0:2] * self.action_scaler[0:2]
        shoulder_action_scaled = self.filtered_action[2:4] * self.action_scaler[2:4]
        leg_action_scaled = self.filtered_action[4:6] * self.action_scaler[4:6]
        wheel_action_scaled = self.filtered_action[6:8] * self.action_scaler[6:8]

        hip_action = self.control_manager.pd_controller(self.kp_hip, hip_action_scaled, pos_hip, self.kd_hip, 0.0, vel_hip)
        shoulder_action = self.control_manager.pd_controller(self.kp_shoulder, shoulder_action_scaled, pos_shoulder, self.kd_shoulder, 0.0, vel_shoulder)
        leg_action = self.control_manager.pd_controller(self.kp_leg, leg_action_scaled, pos_leg, self.kd_leg, 0.0, vel_leg)
        wheel_action = self.control_manager.pd_controller(self.kp_wheel, 0.0, pos_wheel, self.kd_wheel, wheel_action_scaled, vel_wheel)

        self.computed_torques = np.concatenate([hip_action, shoulder_action, leg_action, wheel_action])

        hip_action_clipped = np.clip(hip_action, -self.config['hardware']['joint_max_torque'], self.config['hardware']['joint_max_torque'])
        shoulder_action_clipped = np.clip(shoulder_action, -self.config['hardware']['joint_max_torque'], self.config['hardware']['joint_max_torque'])
        leg_action_clipped = np.clip(leg_action, -self.config['hardware']['joint_max_torque'], self.config['hardware']['joint_max_torque'])
        wheel_action_clipped = np.clip(wheel_action, -self.config['hardware']['wheel_max_torque'], self.config['hardware']['wheel_max_torque'])

        self.applied_torques = np.concatenate([hip_action_clipped, shoulder_action_clipped, leg_action_clipped, wheel_action_clipped])
        self.do_simulation(self.applied_torques, self.frame_skip)

        self.obs = self._get_obs()
        terminated = self._is_done()
        truncated = False
        info = self._get_info()

        self.previous_action = self.action

        return self.obs, terminated, truncated, info

    def _get_info(self):
        q = self.data.qpos[self.q_indices]
        qd = self.data.qvel[self.qd_indices]
        cur_state = [q[0], q[1], q[2], q[3], q[4], q[5], qd[6], qd[7]]
        info = {
            "dt": self.dt_ * self.frame_skip,
            "action": self.action,
            "torque": self.applied_torques,
            "action_diff_RMS": np.linalg.norm(self.action - self.previous_action),
            "lin_vel_x": self.data.sensor('linear-velocity').data.astype(np.float32)[0],
            "lin_vel_y": self.data.sensor('linear-velocity').data.astype(np.float32)[1],
            "ang_vel_z": self.obs[16],
            "pos_z" : self.data.qpos[2],
            "ang_roll" :  self.obs[17],
            "ang_pitch":  self.obs[18],
            "set_points": self.action * self.action_scaler,
            "cur_state": cur_state
        }
        return info

    def _get_reset_info(self):
        return self._get_info()

    def _is_done(self):
        contact_forces = self.data.cfrc_ext[1:12]  # External contact forces
        base_contact = contact_forces[0] > 1.0
        hip_l_contact = contact_forces[1] > 1.0
        hip_r_contact = contact_forces[5] > 1.0
        shoulder_l_contact = contact_forces[2] > 1.0
        shoulder_r_contact = contact_forces[6] > 1.0
        leg_l_contact = contact_forces[3] > 1.0
        leg_r_contact = contact_forces[7] > 1.0
        contact = base_contact.any() or hip_l_contact.any() or hip_r_contact.any() or shoulder_l_contact.any() or shoulder_r_contact.any() or leg_l_contact.any() or leg_r_contact.any()
        return contact

    def reset_model(self):
        self.action = np.zeros(self.action_dim)
        self.previous_action = np.zeros(self.action_dim)
        self.control_manager.reset()
        self.applied_torques = np.zeros(self.action_dim)
        self.user_command = np.zeros(self.command_dim)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos()
        self.data.qvel[:] = 0
        self.obs = self._get_obs()
        return self.obs

    def initial_qpos(self):
        qpos = np.zeros(self.model.nq)
        qpos[2] = 0.40189
        qpos[3:7] = np.array([1, 0, 0, 0])
        qpos[7:15] = np.array([0, 0.0, -0.0, 0, 0, 0.0, -0.0, 0])
        qpos[7:15] = uniform_noisy_data(qpos[7:15], lower=-self.init_noise, upper=self.init_noise)
        return qpos

    def close(self):
        if self.viewer is not None:
            if glfw.get_current_context() == self.viewer.window:
                glfw.make_context_current(None)
            glfw.destroy_window(self.viewer.window)
            glfw.terminate()
            self.viewer = None
            print("Viewer closed")
        super().close()  # Call the parent class's close method to ensure everything is properly closed


