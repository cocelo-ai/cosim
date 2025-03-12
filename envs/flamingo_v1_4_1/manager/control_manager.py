import random
import copy

class ControlManager:
    def __init__(self, config):
        self.prev_action = None
        self.filtered_action = None
        self.prob = config["random"]["action_delay_prob"]

    @staticmethod
    def pd_controller(kp, tq, q, kd, td, d):
        return kp * (tq - q) + kd * (td - d)

    def delay_filter(self, action):
        v = random.uniform(0, 1)
        delay_flag = True if self.prob > v else False
        if not delay_flag or self.prev_action is None:
            self.prev_action = action
            return action
        else:  # when delay_flag is True
            output = copy.deepcopy(self.prev_action)
            self.prev_action = action
            return output

    def reset(self):
        self.prev_action = None
        self.filtered_action = None



