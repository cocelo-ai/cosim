import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from typing import (Tuple, SupportsFloat)


class BaseEnv(ABC):
    """
    Abstract base class for an environment wrapper for the Sim2Sim framework.
    Defines the standard interface that all environment implementations must follow.
    """

    def __init__(self):
        """
        Initializes the environment.
        This constructor does not perform any specific initialization and should be
        overridden by subclasses if necessary.
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state.

        :return:
            - observation (np.ndarray): The initial observation after resetting the environment.
            - info (dict): Additional information about the environment's state, which may
              include metadata or debugging information.
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """
        Executes a step in the environment given an action.

        :param action:
            - np.ndarray: The action to be taken by the agent.

        :return:
            - observation (np.ndarray): The new observation after executing the action.
            - terminated (bool): Whether the episode has ended due to reaching a terminal state.
            - truncated (bool): Whether the episode was truncated due to external constraints
              such as time limits.
            - info (dict): Additional information about the environment’s state, which may
              include diagnostic data or auxiliary variables.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Renders the environment.

        Provides a visual representation of the environment, such as a GUI window
        or textual output. This is useful for debugging and analysis.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the environment and releases resources.

        Ensures proper cleanup, such as closing windows or stopping background processes.
        Should be called when the environment is no longer needed.
        """
        pass


class TimeLimitWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.max_sim_step = int(config["env"]["max_duration"] * 50)
        self.sim_step = 0
        self.reset_flag = False

    def reset(self):
        self.reset_flag = True
        self.sim_step = 0
        init_state, info = self.env.reset()

        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call `reset()` before calling `step()`."
        self.sim_step += 1
        next_state, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.reset_flag = True

        if self.sim_step == self.max_sim_step:
            truncated = True
            self.reset_flag = False

        return next_state, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class ActionInStateWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.state_dim = env.state_dim + config["env"]["action_dim"]
        self.action_dim = env.action_dim
        self.reset_flag = False

    def reset(self):
        self.reset_flag = True
        init_state, info = self.env.reset()
        init_state = np.concatenate((init_state, np.zeros(self.action_dim)))

        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call `reset()` before calling `step()`."
        next_state, terminated, truncated, info = self.env.step(action)
        next_state = np.concatenate((next_state, action))

        if terminated or truncated:
            self.reset_flag = False

        return next_state, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class StateStackWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.num_stack = config["env"]["num_stack"]
        self.state_dim = env.state_dim * self.num_stack
        self.action_dim = env.action_dim
        self.state_stack = np.zeros((self.num_stack, env.state_dim))
        self.reset_flag = False

    def reset(self):
        self.reset_flag = True
        init_state, info = self.env.reset()
        for i in range(self.config["env"]["num_stack"] ):
            self.state_stack[i] = init_state
        init_state = self.state_stack.ravel()

        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call `reset()` before calling `step()`."
        next_state, terminated, truncated, info = self.env.step(action)

        for i in range(self.num_stack - 1):
            self.state_stack[self.num_stack -1 - i] = self.state_stack[self.num_stack - 2 - i]
        self.state_stack[0] = next_state
        next_state = self.state_stack.ravel()

        if terminated or truncated:
            self.reset_flag = False

        return next_state, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class TimeInStateWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.state_dim = env.state_dim + 1
        self.action_dim = env.action_dim
        self.tick = 0.02  # 50Hz
        self.time = 0
        self.reset_flag = False

    def reset(self):
        self.reset_flag = True
        self.time = 0
        init_state, info = self.env.reset()
        init_state = np.concatenate((init_state, np.zeros(1)))

        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call `reset()` before calling `step()`."
        self.time += self.tick
        next_state, terminated, truncated, info = self.env.step(action)
        next_state = np.concatenate((next_state, np.array([self.time])))

        if terminated or truncated:
            self.reset_flag = False

        return next_state, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class CommandWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.state_dim = env.state_dim * config["env"]["command_dim"]
        self.action_dim = env.action_dim
        self.command_dim = config["env"]["command_dim"]
        self.user_command = np.zeros(config["env"]["command_dim"])
        self.reset_flag = False

    def receive_user_command(self, user_command):
        self.user_command = user_command[:self.command_dim]

    def reset(self):
        self.reset_flag = True
        init_state, info = self.env.reset()
        init_state = np.concatenate((init_state, np.zeros(self.config["env"]["command_dim"])))

        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call `reset()` before calling `step()`."
        next_state, terminated, truncated, info = self.env.step(action)
        next_state = np.concatenate((next_state, self.user_command))

        info["lin_vel_x_command"] = self.user_command[0]
        info["lin_vel_y_command"] = self.user_command[1]
        if len(self.user_command) > 2:
            info["ang_vel_z_command"] = self.user_command[2]
        if len(self.user_command) > 3:
            info["pos_z_command"] = self.user_command[3]
        if len(self.user_command) > 4:
            info["ang_roll_command"] = self.user_command[4]
        if len(self.user_command) > 5:
            info["ang_pitch_command"] = self.user_command[5]

        if terminated or truncated:
            self.reset_flag = False

        return next_state, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
