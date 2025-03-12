import mujoco

class MuJoCoUtils:
    def __init__(self, model):
        self.model = model

    def get_body_indices_by_name(self, body_names):
        """
        Get the indices of bodies for given body names.

        Args:
            model: MuJoCo mjModel instance.
            body_names: List of body names to fetch indices for.

        Returns:
            body_indices: List of body indices corresponding to body names.
        """
        body_indices = []
        for body_name in body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                raise ValueError(f"Body name '{body_name}' not found in the model.")
            body_indices.append(body_id)
        return body_indices

    def get_qpos_joint_indices_by_name(self, joint_names):
        """
        Get the indices of qpos and qvel for given joint names.

        Args:
            model: MuJoCo mjModel instance.
            joint_names: List of joint names to fetch indices for.

        Returns:
            qpos_indices: List of qpos indices corresponding to joint names.
            qvel_indices: List of qvel indices corresponding to joint names.
        """
        qpos_indices = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint name '{joint_name}' not found in the model.")
            # Fetch qpos and qvel indices
            qpos_indices.append(self.model.jnt_qposadr[joint_id])
        return qpos_indices


    def get_qvel_joint_indices_by_name(self, joint_names):
        """
        Get the indices of qpos and qvel for given joint names.

        Args:
            model: MuJoCo mjModel instance.
            joint_names: List of joint names to fetch indices for.

        Returns:
            qpos_indices: List of qpos indices corresponding to joint names.
            qvel_indices: List of qvel indices corresponding to joint names.
        """
        qvel_indices = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint name '{joint_name}' not found in the model.")
            # Fetch qpos and qvel indices
            qvel_indices.append(self.model.jnt_dofadr[joint_id])
        return qvel_indices