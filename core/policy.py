import onnxruntime as ort
import numpy as np


class MLPPolicy:
    def __init__(self, policy_path):
        self.ort_session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [output.name for output in self.ort_session.get_outputs()]

    def get_action(self, state: np.ndarray):
        state = state.astype(np.float32)
        try:
            _state = np.expand_dims(state, axis=0)
            action = self.ort_session.run(self.output_names, {self.input_name: _state})[0]
            action = np.squeeze(action, axis=0)
        except:
            action = self.ort_session.run(self.output_names, {self.input_name: state})[0]
        action = np.clip(action, -1, 1)
        return action


class LSTMPolicy:
    def __init__(self, config, policy_path):
        self.ort_session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self.input_names = [self.ort_session.get_inputs()[0].name, "h_in", "c_in"]
        assert self.ort_session.get_inputs()[1].name == "h_in" and self.ort_session.get_inputs()[2].name == "c_in",\
            "The input names of ONNX policy must include 'h_in' and 'c_in'"

        self.h_in = np.zeros((1, 1, config["policy"]["h_in_dim"]), dtype=np.float32)
        self.c_in = np.zeros((1, 1, config["policy"]["c_in_dim"]), dtype=np.float32)

    def get_action(self, state):
        state = state.astype(np.float32)
        state = np.expand_dims(state, axis=0)
        policy_input = {self.input_names[0]: state,
                 "h_in": self.h_in,
                 "c_in": self.c_in,
                 }
        action, h_out, c_out = self.ort_session.run(None, policy_input)
        self.h_in = h_out
        self.c_in = c_out

        action = np.squeeze(action, axis=0)
        action = np.clip(action, -1, 1)
        return action

def build_policy(config, policy_path):
    if config["policy"]["use_lstm"]:
        return LSTMPolicy(config, policy_path)
    else:
        return MLPPolicy(policy_path)