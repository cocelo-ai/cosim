import os
import time
import numpy as np
from core.policy import build_policy
from core.reporter import Reporter
from envs.build import build_env
from PyQt5.QtCore import QObject, pyqtSignal

class Tester(QObject):
    # 테스트 종료 시그널
    finished = pyqtSignal()
    # 각 step 종료 시그널 (MainWindow에 알림)
    stepFinished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.user_command = None
        self._stop = False  # 중단 플래그

    def load_config(self, config):
        self.config = config

    def load_policy(self, policy_path):
        self.policy_path = policy_path

    def init_user_command(self):
        """테스트 시작 전에 사용자 명령어 배열을 초기화합니다."""
        self.user_command = np.zeros(self.config["env"]["command_dim"])

    def receive_user_command(self):
        """
        현재의 사용자 명령어 값을 환경에 전달합니다.
        """
        if self.user_command is None:
            self.init_user_command()
        self.env.receive_user_command(self.user_command)

    def update_command(self, index, value):
        """
        UI에서 호출되어 특정 인덱스의 command 값을 업데이트합니다.
        """
        if self.user_command is None:
            self.init_user_command()
        if index < self.config["env"]["command_dim"]:
            self.user_command[index] = value

    def test(self):
        # 환경, 정책, 리포터 생성
        self.policy = build_policy(self.config, policy_path=os.path.join(self.policy_path))
        report_path = os.path.join(os.path.dirname(self.policy_path), 'report.pdf')
        self.reporter = Reporter(report_path=report_path, config=self.config)

        self.env = build_env(self.config)
        state, info = self.env.reset()
        done = False
   
        # 테스트 루프
        while not done and not self._stop:
            self.env.render()
            # UI에서 업데이트된 command 값을 환경에 반영
            self.receive_user_command()
            action = self.policy.get_action(state)

            assert self.user_command is not None, "user_command must not be None."
            next_state, terminated, truncated, info = self.env.step(action)

            # 리포터에 정보 기록 및 step 종료 시그널 emit
            self.reporter.write_info(info)
            self.stepFinished.emit()

            done = terminated or truncated
            state = next_state

        # 테스트 종료 후 정리
        self.reporter.generate_report()
        self.close()
        self.finished.emit()

    def stop(self):
        """테스트 루프를 중단시키는 함수입니다."""
        self._stop = True

    def close(self):
        """환경 종료를 시도합니다."""
        try:
            self.env.close()
        except Exception:
            pass
