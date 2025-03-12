import os
import sys
import yaml
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QMessageBox, QMainWindow,
    QFileDialog, QGroupBox, QScrollArea, QComboBox, QLineEdit, QSlider, QApplication
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt, QEvent, QUrl
from PyQt5.QtGui import QDesktopServices, QIcon
from core.tester import Tester


# Custom QComboBox that ignores mouse wheel events
class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


# Custom QSlider that ignores mouse wheel events
class NoWheelSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()


# Button that ignores mouse clicks (only responds to keyboard input)
class NonClickableButton(QPushButton):
    def mousePressEvent(self, event):
        event.ignore()

    def mouseReleaseEvent(self, event):
        event.ignore()


class TesterWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, tester: Tester):
        super().__init__()
        self.tester = tester

    def run(self):
        try:
            self.tester.init_user_command()
            self.tester.test()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        cur_file_path = os.path.abspath(__file__)
        config_path = os.path.join(os.path.dirname(cur_file_path), "../config/env_table.yaml")
        config_path = os.path.abspath(config_path)
        with open(config_path) as f:
            self.env_config = yaml.full_load(f)

        self._init_window()
        self._init_variables()
        self._setup_ui()
        self._init_default_command_values()
        self.status_label.setText("대기 중")
        self.env_id_cb.currentTextChanged.connect(self.update_defaults)
        self.update_defaults(self.env_id_cb.currentText())

    def _init_window(self):
        app_logo_path = os.path.join(os.path.dirname(__file__), "icon", "main_logo_128_128.png")
        self.setWindowIcon(QIcon(app_logo_path))
        self.setWindowTitle("cosim  -  v1.0")
        self.resize(950, 1000)
        # 기본적으로 메인 윈도우에 이벤트 필터를 설치
        self.installEventFilter(self)

    def _init_variables(self):
        self.key_mapping = {}
        self.active_keys = {}
        self.thread = None
        self.worker = None
        self.tester = None
        self.current_command_values = [0.0] * 6

        # Lists for command-related QLineEdit/QLabel widgets
        self.command_sensitivity_le_list = []
        self.max_command_value_le_list = []
        self.command_initial_value_le_list = []
        self.command_timer = None

    def _init_default_command_values(self):
        try:
            self.current_command_values = [float(widget.text()) for widget in self.command_initial_value_le_list]
        except Exception:
            self.current_command_values = [0.0] * 6

    def update_defaults(self, new_env_id):
        settings = self.env_config.get(new_env_id)
        # Update hardware settings
        self.Kp_hip.setText(settings["Kp_hip"])
        self.Kp_shoulder.setText(settings["Kp_shoulder"])
        self.Kp_leg.setText(settings["Kp_leg"])
        self.Kp_wheel.setText(settings["Kp_wheel"])
        self.Kd_hip.setText(settings["Kd_hip"])
        self.Kd_shoulder.setText(settings["Kd_shoulder"])
        self.Kd_leg.setText(settings["Kd_leg"])
        self.Kd_wheel.setText(settings["Kd_wheel"])
        self.joint_max_torque_le.setText(settings["joint_max_torque"])
        self.wheel_max_torque_le.setText(settings["wheel_max_torque"])
        # command[3]의 초기값은 환경에 따라 갱신
        if isinstance(self.command_initial_value_le_list[3], QLineEdit):
            self.command_initial_value_le_list[3].setText(settings["command_3_initial"])

    def showEvent(self, event):
        self.centralWidget().setFocus()
        super().showEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            self.handle_key_press(event)
            return True
        elif event.type() == QEvent.KeyRelease:
            self.handle_key_release(event)
            return True
        return super().eventFilter(obj, event)

    def handle_key_press(self, event):
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in self.key_mapping and key not in self.active_keys:
            btn, cmd_index, direction = self.key_mapping[key]
            btn.setChecked(True)
            self.active_keys[key] = {"cmd_index": cmd_index, "direction": direction}

    def handle_key_release(self, event):
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in self.key_mapping:
            btn, cmd_index, _ = self.key_mapping[key]
            btn.setChecked(False)
            if key in self.active_keys:
                self.active_keys.pop(key)
            default_value = self._get_default_command_value(cmd_index)
            self.current_command_values[cmd_index] = default_value
            self._update_command_button(cmd_index, default_value)

    def _get_default_command_value(self, index):
        try:
            return float(self.command_initial_value_le_list[index].text())
        except Exception:
            return 0.0

    def _update_status_label(self):
        html_text = (
            "<html><head><style>"
            "h3 { margin: 0 0 8px 0; }"
            "table { border-collapse: collapse; }"
            "td { padding: 4px 8px; border: 1px solid #ddd; }"
            "</style></head><body>"
            "<h4> Current Command Values</h4><table>"
        )
        for i, value in enumerate(self.current_command_values):
            if i % 6 == 0:
                if i != 0:
                    html_text += "</tr>"
                html_text += "<tr>"
            html_text += f"<td>[{i}] = {value:.3f}</td>"
        html_text += "</tr></table></body></html>"
        self.status_label.setText(html_text)

    def _update_command_button(self, index, value):
        self.current_command_values[index] = value
        self._update_status_label()

    def send_current_command(self):
        for key_info in self.active_keys.values():
            cmd_index = key_info["cmd_index"]
            direction = key_info["direction"]
            step = self._parse_float(self.command_sensitivity_le_list[cmd_index].text(), 0.1)
            max_command_value = self._parse_float(self.max_command_value_le_list[cmd_index].text(), 2.0)
            current_value = self.current_command_values[cmd_index]
            new_value = current_value + direction * step
            new_value = min(new_value, max_command_value) if direction > 0 else max(new_value, -max_command_value)
            self.current_command_values[cmd_index] = new_value
            self._update_command_button(cmd_index, new_value)

        if self.tester:
            for i, value in enumerate(self.current_command_values):
                self.tester.update_command(i, value)
        self._update_status_label()

    def _parse_float(self, text, default):
        try:
            return float(text)
        except Exception:
            return default

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Top area: Left (configuration) and right (command settings/input)
        top_h_layout = QHBoxLayout()
        top_h_layout.setSpacing(15)
        main_layout.addLayout(top_h_layout)

        # Left: Configuration settings in a scrollable area
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        top_h_layout.addWidget(config_scroll, 3)
        config_widget = QWidget()
        config_scroll.setWidget(config_widget)
        self.config_layout = QVBoxLayout(config_widget)
        self.config_layout.setContentsMargins(10, 10, 10, 10)
        self.config_layout.setSpacing(15)
        self._create_top_config_groups()
        self._create_hardware_group()
        self._create_random_group()

        # Right: Command settings and key input visual buttons
        right_v_layout = QVBoxLayout()
        top_h_layout.addLayout(right_v_layout, 1)
        self._create_command_settings_group(right_v_layout)
        self._setup_key_visual_buttons(right_v_layout)

        self.status_label = QLabel("대기 중")
        self.status_label.setStyleSheet("font-size: 14px;")
        main_layout.addWidget(self.status_label)

        # Bottom: Start/Stop buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.start_button = QPushButton("Start Test")
        self.start_button.setFixedWidth(120)
        self.start_button.clicked.connect(self.start_test)
        btn_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop Test")
        self.stop_button.setFixedWidth(120)
        self.stop_button.clicked.connect(self.stop_test)
        self.stop_button.setEnabled(False)
        btn_layout.addWidget(self.stop_button)
        main_layout.addLayout(btn_layout)

        self._apply_styles()

    def _create_top_config_groups(self):
        top_layout = QHBoxLayout()
        top_layout.setSpacing(15)
        self.config_layout.addLayout(top_layout)
        self._create_env_group(top_layout)
        self._create_policy_group(top_layout)  # Policy 그룹에 ONNX 파일 선택 포함

    def _create_env_group(self, parent_layout):
        env_group = QGroupBox("Environment Settings")
        env_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        env_layout = QFormLayout()
        env_layout.setLabelAlignment(Qt.AlignRight)
        env_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        env_layout.setSpacing(8)
        env_group.setLayout(env_layout)

        self.env_id_cb = NoWheelComboBox()
        self.env_id_cb.addItems(self.env_config.keys())
        self.env_id_cb.setCurrentText("flamingo_v1_4_1")
        env_layout.addRow("ID:", self.env_id_cb)

        self.terrain_id_cb = NoWheelComboBox()
        self.terrain_id_cb.addItems(['flat', 'rocky_easy', 'rocky_hard', 'slope_easy', 'slope_hard', 'stairs_easy', 'stairs_hard'])
        self.terrain_id_cb.setCurrentText("flat")
        env_layout.addRow("Terrain:", self.terrain_id_cb)

        self.max_duration_le = QLineEdit("180.0")
        env_layout.addRow("Max Duration:", self.max_duration_le)
        self.observation_dim_le = QLineEdit("20")
        env_layout.addRow("Observation Dim:", self.observation_dim_le)
        self.command_dim_le = QLineEdit("6")
        env_layout.addRow("Command Dim:", self.command_dim_le)
        self.action_dim_le = QLineEdit("8")
        env_layout.addRow("Action Dim:", self.action_dim_le)

        self.action_in_state_cb = NoWheelComboBox()
        self.action_in_state_cb.addItems(["True", "False"])
        self.action_in_state_cb.setCurrentText("True")
        env_layout.addRow("Action in State:", self.action_in_state_cb)

        self.time_in_state_cb = NoWheelComboBox()
        self.time_in_state_cb.addItems(["True", "False"])
        self.time_in_state_cb.setCurrentText("False")
        env_layout.addRow("Time in State:", self.time_in_state_cb)

        self.num_stack_cb = NoWheelComboBox()
        self.num_stack_cb.addItems([str(i) for i in range(1, 16)])
        self.num_stack_cb.setCurrentText("3")
        env_layout.addRow("Num Stack:", self.num_stack_cb)

        parent_layout.addWidget(env_group, 1)


    def _create_policy_group(self, parent_layout):
        policy_group = QGroupBox("Policy Settings")
        policy_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        policy_layout = QFormLayout()
        policy_layout.setLabelAlignment(Qt.AlignRight)
        policy_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        policy_layout.setSpacing(8)
        policy_group.setLayout(policy_layout)

        self.use_lstm_cb = NoWheelComboBox()
        self.use_lstm_cb.addItems(["True", "False"])
        self.use_lstm_cb.setCurrentText("False")
        policy_layout.addRow("Use LSTM:", self.use_lstm_cb)

        self.h_in_dim_le = QLineEdit("256")
        policy_layout.addRow("h_in Dim:", self.h_in_dim_le)
        self.c_in_dim_le = QLineEdit("256")
        policy_layout.addRow("c_in Dim:", self.c_in_dim_le)

        # ONNX 파일 선택 부분 추가 (이전 _create_policy_file_group() 대신)
        self.policy_file_le = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_policy_file)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.policy_file_le)
        file_layout.addWidget(browse_btn)
        policy_layout.addRow("ONNX File:", file_layout)

        parent_layout.addWidget(policy_group, 1)

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit

    def _create_hardware_group(self):
        hardware_group = QGroupBox("Hardware Settings")
        hardware_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 5px; padding: 2px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 5px; padding: 0 2px; }"
        )

        # 최상위 레이아웃 (수직)
        v_layout_main = QVBoxLayout()
        v_layout_main.setContentsMargins(2, 2, 2, 2)
        v_layout_main.setSpacing(10)
        # 전체를 왼쪽 정렬
        v_layout_main.setAlignment(Qt.AlignLeft)
        hardware_group.setLayout(v_layout_main)

        settings = self.env_config.get(self.env_id_cb.currentText(), {})

        # -------------------- (1) 상단: 관절별 P, D 게인 설정 --------------------
        gains_layout = QHBoxLayout()
        gains_layout.setContentsMargins(2, 2, 2, 2)
        gains_layout.setSpacing(10)
        # 왼쪽 정렬
        gains_layout.setAlignment(Qt.AlignLeft)
        v_layout_main.addLayout(gains_layout)

        joints = [
            ("Hip", settings.get("Kp_hip", "0.0"), settings.get("Kd_hip", "0.0")),
            ("Shoulder", settings.get("Kp_shoulder", "0.0"), settings.get("Kd_shoulder", "0.0")),
            ("Leg", settings.get("Kp_leg", "0.0"), settings.get("Kd_leg", "0.0")),
            ("Wheel", settings.get("Kp_wheel", "0.0"), settings.get("Kd_wheel", "0.0"))
        ]

        def create_line_edit(default_value):
            le = QLineEdit(str(default_value))
            le.setFixedWidth(45)
            # 입력값을 왼쪽 정렬
            le.setAlignment(Qt.AlignLeft)
            font = le.font()
            font.setPointSize(6)
            le.setFont(font)
            return le

        for joint_name, kp, kd in joints:
            joint_widget = QWidget()
            v_layout = QVBoxLayout(joint_widget)
            v_layout.setContentsMargins(1, 1, 1, 1)
            v_layout.setSpacing(2)
            # 왼쪽 정렬
            v_layout.setAlignment(Qt.AlignLeft)

            # 관절명 라벨
            joint_label = QLabel(joint_name)
            # 라벨 왼쪽 정렬
            joint_label.setAlignment(Qt.AlignLeft)
            font_label = joint_label.font()
            font_label.setPointSize(6)
            joint_label.setFont(font_label)
            v_layout.addWidget(joint_label)

            # P, D 게인 부분
            gains_sub_layout = QHBoxLayout()
            gains_sub_layout.setContentsMargins(1, 1, 1, 1)
            gains_sub_layout.setSpacing(2)
            # 왼쪽 정렬
            gains_sub_layout.setAlignment(Qt.AlignLeft)

            p_label = QLabel("P:")
            p_label.setAlignment(Qt.AlignLeft)
            p_label.setFont(font_label)
            p_gain = create_line_edit(kp)

            d_label = QLabel("D:")
            d_label.setAlignment(Qt.AlignLeft)
            d_label.setFont(font_label)
            d_gain = create_line_edit(kd)

            gains_sub_layout.addWidget(p_label)
            gains_sub_layout.addWidget(p_gain)
            gains_sub_layout.addWidget(d_label)
            gains_sub_layout.addWidget(d_gain)
            v_layout.addLayout(gains_sub_layout)

            gains_layout.addWidget(joint_widget)

            # 인스턴스 변수에 저장 (필요 시 사용)
            if joint_name.lower() == "hip":
                self.Kp_hip, self.Kd_hip = p_gain, d_gain
            elif joint_name.lower() == "shoulder":
                self.Kp_shoulder, self.Kd_shoulder = p_gain, d_gain
            elif joint_name.lower() == "leg":
                self.Kp_leg, self.Kd_leg = p_gain, d_gain
            elif joint_name.lower() == "wheel":
                self.Kp_wheel, self.Kd_wheel = p_gain, d_gain

        # -------------------- (2) 하단: Joint Max Torque, Wheel Max Torque --------------------
        torque_layout = QHBoxLayout()
        torque_layout.setContentsMargins(2, 2, 2, 2)
        torque_layout.setSpacing(15)
        # 왼쪽 정렬
        torque_layout.setAlignment(Qt.AlignLeft)

        # Joint Max Torque
        joint_max_torque_label = QLabel("Joint Max Torque:")
        # 라벨 왼쪽 정렬
        joint_max_torque_label.setAlignment(Qt.AlignLeft)
        joint_max_torque_label.setStyleSheet("font-size: 12px;")
        joint_max_torque_value = settings.get("joint_max_torque", "100")
        self.joint_max_torque_le = create_line_edit(joint_max_torque_value)

        torque_layout.addWidget(joint_max_torque_label)
        torque_layout.addWidget(self.joint_max_torque_le)

        # Wheel Max Torque
        wheel_max_torque_label = QLabel("Wheel Max Torque:")
        # 라벨 왼쪽 정렬
        wheel_max_torque_label.setAlignment(Qt.AlignLeft)
        wheel_max_torque_label.setStyleSheet("font-size: 12px;")
        wheel_max_torque_value = settings.get("wheel_max_torque", "100")
        self.wheel_max_torque_le = create_line_edit(wheel_max_torque_value)

        torque_layout.addWidget(wheel_max_torque_label)
        torque_layout.addWidget(self.wheel_max_torque_le)

        v_layout_main.addLayout(torque_layout)

        # 완성된 groupBox를 config 레이아웃에 추가
        self.config_layout.addWidget(hardware_group)

    def _create_random_group(self):
        random_group = QGroupBox("Random Settings")
        random_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form_layout.setSpacing(8)
        random_group.setLayout(form_layout)

        self.precision_cb = NoWheelComboBox()
        self.precision_cb.addItems(["low", "medium", "high", "ultra", "extreme"])
        self.precision_cb.setCurrentText("medium")
        form_layout.addRow("Precision:", self.precision_cb)

        self.sensor_noise_cb = NoWheelComboBox()
        self.sensor_noise_cb.addItems(["none", "low", "medium", "high", "ultra", "extreme"])
        self.sensor_noise_cb.setCurrentText("low")
        form_layout.addRow("Sensor Noise:", self.sensor_noise_cb)

        def create_slider_row(slider, min_val, max_val, init_val, scale, decimals):
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(init_val)
            value_label = QLabel(f"{init_val / scale:.{decimals}f}")
            slider.valueChanged.connect(lambda v: value_label.setText(f"{v / scale:.{decimals}f}"))
            h_layout = QHBoxLayout()
            h_layout.addWidget(slider)
            h_layout.addWidget(value_label)
            return h_layout

        self.init_noise_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Init Noise:", create_slider_row(self.init_noise_slider, 0, 100, 5, 100, 2))
        self.sliding_friction_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Sliding Friction:", create_slider_row(self.sliding_friction_slider, 0, 100, 80, 100, 2))
        self.torsional_friction_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Torsional Friction:", create_slider_row(self.torsional_friction_slider, 0, 10, 2, 100, 2))
        self.rolling_friction_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Rolling Friction:", create_slider_row(self.rolling_friction_slider, 0, 10, 1, 100, 2))
        self.friction_loss_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Friction Loss:", create_slider_row(self.friction_loss_slider, 0, 100, 10, 100, 2))
        self.action_delay_prob_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Action Delay Prob.:", create_slider_row(self.action_delay_prob_slider, 0, 100, 5, 100, 2))
        self.mass_noise_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Mass Noise:", create_slider_row(self.mass_noise_slider, 0, 50, 5, 100, 2))
        self.load_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Load:", create_slider_row(self.load_slider, 0, 200, 0, 10, 1))
        self.config_layout.addWidget(random_group)

    def _create_command_settings_group(self, parent_layout):
        command_group = QGroupBox("Command Settings")
        command_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        grid_layout = QGridLayout(command_group)
        grid_layout.addWidget(QLabel("Index"), 0, 0)
        grid_layout.addWidget(QLabel("Sensitivity"), 0, 1)
        grid_layout.addWidget(QLabel("Max Value"), 0, 2)
        grid_layout.addWidget(QLabel("Initial Value"), 0, 3)

        settings = self.env_config.get(self.env_id_cb.currentText())
        for i in range(6):
            label = QLabel(f"command[{i}]")
            sensitivity_le = QLineEdit("0.02")
            max_value_le = QLineEdit("1.5" if i in [0, 1, 2] else "1")
            init_value_widget = QLineEdit(settings["command_3_initial"]) if i == 3 else QLabel("0.0")

            grid_layout.addWidget(label, i + 1, 0)
            grid_layout.addWidget(sensitivity_le, i + 1, 1)
            grid_layout.addWidget(max_value_le, i + 1, 2)
            grid_layout.addWidget(init_value_widget, i + 1, 3)

            self.command_sensitivity_le_list.append(sensitivity_le)
            self.max_command_value_le_list.append(max_value_le)
            self.command_initial_value_le_list.append(init_value_widget)
        parent_layout.addWidget(command_group)

    def _setup_key_visual_buttons(self, parent_layout):
        button_style = (
            "NonClickableButton { background-color: #3C3F41; border: none; color: #FFFFFF; "
            "font-size: 11px; padding: 10px; border-radius: 10px; min-width: 50px; min-height: 50px; }"
            "NonClickableButton:checked { background-color: #4E94D4; }"
        )
        key_group = QGroupBox("Command Input")
        key_layout = QVBoxLayout(key_group)
        key_layout.setSpacing(10)

        # Direction keys (W, A, S, D)
        dir_group = QGroupBox("command[0], command[2]")
        dir_layout = QGridLayout(dir_group)
        self.btn_up = NonClickableButton("W")
        self.btn_up.setStyleSheet(button_style)
        self.btn_up.setCheckable(True)
        dir_layout.addWidget(self.btn_up, 0, 1)
        self.btn_left = NonClickableButton("A")
        self.btn_left.setStyleSheet(button_style)
        self.btn_left.setCheckable(True)
        dir_layout.addWidget(self.btn_left, 1, 0)
        self.btn_right = NonClickableButton("D")
        self.btn_right.setStyleSheet(button_style)
        self.btn_right.setCheckable(True)
        dir_layout.addWidget(self.btn_right, 1, 2)
        self.btn_down = NonClickableButton("S")
        self.btn_down.setStyleSheet(button_style)
        self.btn_down.setCheckable(True)
        dir_layout.addWidget(self.btn_down, 1, 1)
        key_layout.addWidget(dir_group)

        # Other keys (I, O, P, J, K, L)
        other_group = QGroupBox("command[3], command[4], command[5]")
        other_layout = QGridLayout(other_group)
        self.btn_i = NonClickableButton("I")
        self.btn_i.setStyleSheet(button_style)
        self.btn_i.setCheckable(True)
        other_layout.addWidget(self.btn_i, 0, 0)
        self.btn_o = NonClickableButton("O")
        self.btn_o.setStyleSheet(button_style)
        self.btn_o.setCheckable(True)
        other_layout.addWidget(self.btn_o, 0, 1)
        self.btn_p = NonClickableButton("P")
        self.btn_p.setStyleSheet(button_style)
        self.btn_p.setCheckable(True)
        other_layout.addWidget(self.btn_p, 0, 2)
        self.btn_j = NonClickableButton("J")
        self.btn_j.setStyleSheet(button_style)
        self.btn_j.setCheckable(True)
        other_layout.addWidget(self.btn_j, 1, 0)
        self.btn_k = NonClickableButton("K")
        self.btn_k.setStyleSheet(button_style)
        self.btn_k.setCheckable(True)
        other_layout.addWidget(self.btn_k, 1, 1)
        self.btn_l = NonClickableButton("L")
        self.btn_l.setStyleSheet(button_style)
        self.btn_l.setCheckable(True)
        other_layout.addWidget(self.btn_l, 1, 2)
        key_layout.addWidget(other_group)

        # ZX group (command[1])
        zx_group = QGroupBox("command[1]")
        zx_layout = QHBoxLayout(zx_group)
        zx_style = (
            "NonClickableButton { background-color: #3C3F41; border: none; color: #FFFFFF; "
            "font-size: 11px; padding: 4px; border-radius: 10px; min-width: 30px; min-height: 30px; }"
            "NonClickableButton:checked { background-color: #4E94D4; }"
        )
        self.btn_z = NonClickableButton("Z")
        self.btn_z.setStyleSheet(zx_style)
        self.btn_z.setCheckable(True)
        zx_layout.addWidget(self.btn_z)
        self.btn_x = NonClickableButton("X")
        self.btn_x.setStyleSheet(zx_style)
        self.btn_x.setCheckable(True)
        zx_layout.addWidget(self.btn_x)
        key_layout.addWidget(zx_group)
        parent_layout.addWidget(key_group, 1)

        # Set key mapping for command value adjustments
        self.key_mapping = {
            Qt.Key_W: (self.btn_up, 0, +1.0),
            Qt.Key_S: (self.btn_down, 0, -1.0),
            Qt.Key_A: (self.btn_left, 2, +1.0),
            Qt.Key_D: (self.btn_right, 2, -1.0),
            Qt.Key_Z: (self.btn_z, 1, -1.0),
            Qt.Key_X: (self.btn_x, 1, +1.0),
            Qt.Key_I: (self.btn_i, 3, +1.0),
            Qt.Key_J: (self.btn_j, 3, -1.0),
            Qt.Key_O: (self.btn_o, 4, +1.0),
            Qt.Key_K: (self.btn_k, 4, -1.0),
            Qt.Key_P: (self.btn_p, 5, +1.0),
            Qt.Key_L: (self.btn_l, 5, -1.0)
        }

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QLineEdit, QComboBox, QSlider {
                padding: 4px;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #4E94D4;
            }
            QPushButton:disabled {
                background-color: #A0A0A0;
            }
            QPushButton:hover:!disabled {
                background-color: #005999;
            }
        """)

    def browse_policy_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Policy ONNX File", os.path.join(os.getcwd(), "weights"),
            "ONNX Files (*.onnx)"
        )
        if file_path:
            self.policy_file_le.setText(file_path)

    def start_test(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("테스트 실행 중...")
        self._update_status_label()

        config = self._gather_config()
        if config is None:
            return

        policy_file_path = self.policy_file_le.text().strip()
        if not policy_file_path or not os.path.isfile(policy_file_path):
            QMessageBox.critical(self, "Error", "유효한 ONNX 파일을 선택해주세요.")
            self._reset_ui_after_test()
            return

        self.tester = Tester()
        self.tester.load_config(config)
        self.tester.load_policy(policy_file_path)
        self._init_default_command_values()
        for i, value in enumerate(self.current_command_values):
            self.tester.update_command(i, value)
        self.tester.stepFinished.connect(self.send_current_command)

        self.thread = QThread()
        self.worker = TesterWorker(self.tester)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_test_finished)
        self.worker.error.connect(self.on_test_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _gather_config(self):
        try:
            config = {
                "env": {
                    "id": self.env_id_cb.currentText(),
                    "terrain": self.terrain_id_cb.currentText(),
                    "action_in_state": self.action_in_state_cb.currentText() == "True",
                    "time_in_state": self.time_in_state_cb.currentText() == "True",
                    "max_duration": float(self.max_duration_le.text().strip()),
                    "observation_dim": int(self.observation_dim_le.text().strip()),
                    "command_dim": int(self.command_dim_le.text().strip()),
                    "action_dim": int(self.action_dim_le.text().strip()),
                    "num_stack": int(self.num_stack_cb.currentText()),
                },
                "policy": {
                    "use_lstm": self.use_lstm_cb.currentText() == "True",
                    "h_in_dim": int(self.h_in_dim_le.text().strip()),
                    "c_in_dim": int(self.c_in_dim_le.text().strip()),
                    "onnx_file": os.path.basename(self.policy_file_le.text())
                },
                "random": {
                    "precision": self.precision_cb.currentText(),
                    "sensor_noise": self.sensor_noise_cb.currentText(),
                    "init_noise": self.init_noise_slider.value() / 100.0,
                    "sliding_friction": self.sliding_friction_slider.value() / 100.0,
                    "torsional_friction": self.torsional_friction_slider.value() / 100.0,
                    "rolling_friction": self.rolling_friction_slider.value() / 100.0,
                    "friction_loss": self.friction_loss_slider.value() / 100.0,
                    "action_delay_prob": self.action_delay_prob_slider.value() / 100.0,
                    "mass_noise": self.mass_noise_slider.value() / 100.0,
                    "load": self.load_slider.value() / 10.0
                },
                "hardware": {
                    "Kp_hip": float(self.Kp_hip.text().strip()),
                    "Kp_shoulder": float(self.Kp_shoulder.text().strip()),
                    "Kp_leg": float(self.Kp_leg.text().strip()),
                    "Kd_hip": float(self.Kd_hip.text().strip()),
                    "Kd_shoulder": float(self.Kd_shoulder.text().strip()),
                    "Kd_leg": float(self.Kd_leg.text().strip()),
                    "Kp_wheel": float(self.Kp_wheel.text().strip()),
                    "Kd_wheel": float(self.Kd_wheel.text().strip()),
                    "joint_max_torque": float(self.joint_max_torque_le.text().strip()),
                    "wheel_max_torque": float(self.wheel_max_torque_le.text().strip())
                }
            }
            cur_file_path = os.path.abspath(__file__)
            config_path = os.path.join(os.path.dirname(cur_file_path), "../config/random_table.yaml")
            config_path = os.path.abspath(config_path)
            with open(config_path) as f:
                random_config = yaml.full_load(f)
            config["random_table"] = random_config["random_table"]
            return config
        except Exception as e:
            QMessageBox.critical(self, "Error", f"파라미터 설정 오류: {e}")
            self._reset_ui_after_test()
            return None

    def _reset_ui_after_test(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("대기 중")

    def reset_command_buttons(self):
        for key in list(self.active_keys.keys()):
            btn, cmd_index, _ = self.key_mapping[key]
            btn.setChecked(False)
            default_value = self._get_default_command_value(cmd_index)
            self._update_command_button(cmd_index, default_value)
            self.active_keys.pop(key)

    def on_test_finished(self):
        self.reset_command_buttons()
        self.status_label.setText("테스트 완료")
        self._reset_ui_after_test()
        reply = QMessageBox.question(
            self,
            "Report 확인",
            "테스트가 종료되었습니다. 리포트를 열람하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            policy_file_path = self.policy_file_le.text().strip()
            report_path = os.path.join(os.path.dirname(policy_file_path), "report.pdf")
            if os.path.isfile(report_path):
                QDesktopServices.openUrl(QUrl.fromLocalFile(report_path))
            else:
                QMessageBox.warning(self, "Warning", "리포트 파일(report.pdf)이 존재하지 않습니다.")

    def on_test_error(self, error_msg):
        QMessageBox.critical(self, "Test Error", error_msg)
        self.status_label.setText("오류 발생")
        self._reset_ui_after_test()

    def stop_test(self):
        if self.tester:
            try:
                self.tester.stop()
                self.status_label.setText("테스트 중지 요청")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"테스트 중지 오류: {e}")
        self.reset_command_buttons()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
