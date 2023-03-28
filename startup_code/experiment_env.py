#! /usr/bin/env python3

import os, sys
import numpy as np
import threading as th
import pandas as pd

from enum import Enum
from basic_api_functions import *


class ControlModes(Enum):
    JOINT_POSITION = 0
    JOINT_VELOCITY = 1
    END_EFFECTOR_POSE = 2
    END_EFFECTOR_TWIST = 3

class RealWorldTrajectory:
    """ A sample real-world trajectory for experiments.
    """

    def __init__(self):
        self.start = {'linear_x': 0.024,
                      'linear_y': -0.341,
                      'linear_z': 0.028,
                      'angular_x': 3.918,
                      'angular_y': 176.9,
                      'angular_z': 6.594}

        self.goal = {'linear_x': -0.0,
                     'linear_y': 0.3,
                     'linear_z': 0.1,
                     'angular_x': -3.1,
                     'angular_y': 177.9,
                     'angular_z': 21.7}

        self.pos_dataset = pd.DataFrame(columns=['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])
        self.twist_dataset = pd.DataFrame(columns=['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])

        self.capture: bool = False

    def capture_data(self, basecyclic: BaseCyclicClient):
        """Log feedback data for a base cyclic client.

        Args:
            basecyclic (BaseCyclicClient): base cyclic object is used for getting feedback from the arm.
        """
        # capturing data is not necessary, unless called upon
        while not self.capture:
            time.sleep(0.01)
            continue

        n_samples = 0
        while self.capture:
            twist_dict = endeffector_twist_feedback(basecyclic)
            pos_dict = endeffector_pose_feedback(basecyclic)

            self.pos_dataset = pd.concat([self.pos_dataset, pd.DataFrame(pos_dict, index=[n_samples])])
            self.twist_dataset = pd.concat([self.twist_dataset, pd.DataFrame(twist_dict, index=[n_samples])])

            time.sleep(0.1)
            n_samples += 1

        print(f'Terminating demonstration data logger with {n_samples} samples')

    def clear_data(self):
        """ Clear datasets to recapture another sequence.
        """

        self.pos_dataset = pd.DataFrame(columns=['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])
        self.twist_dataset = pd.DataFrame(columns=['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])

    def save_data(self, dir: str = os.getcwd()):
        save_dir = os.path.join(dir, 'dems')
        os.makedirs(save_dir, exist_ok=True)
        self.pos_dataset.to_csv(path_or_buf=os.path.join(save_dir, 'pos.csv'))
        self.twist_dataset.to_csv(path_or_buf=os.path.join(save_dir, 'twist.csv'))


class KinovaDSExperiments:
    def __init__(self, mode=ControlModes.END_EFFECTOR_POSE, device_ip: str = '192.168.1.10',
                 device_port: int = 10000, username: str = 'admin', password: str = 'admin',
                 session_inactivity_timeout: int = 60000, connection_inactivity_timeout: int = 20000):

        # build the transport layer
        self.__transport = TCPTransport() if device_port == DeviceConnection.TCP_PORT else UDPTransport()
        self.__router = RouterClient(self.__transport, RouterClient.basicErrorCallback)
        self.__transport.connect(device_ip, device_port)
        self.__trajectory_handle = RealWorldTrajectory()

        # create a session
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = username
        session_info.password = password
        session_info.session_inactivity_timeout = session_inactivity_timeout
        session_info.connection_inactivity_timeout = connection_inactivity_timeout

        self.__session_manager = SessionManager(self.__router)
        print(f'Logging as {session_info.username} on device {device_ip}')
        self.__session_manager.CreateSession(session_info)

        # create base and basecyclic objects
        self.__base = BaseClient(self.__router)
        self.__basecyclic = BaseCyclicClient(self.__router)
        print(f'Initialization complete')

        # start the logger
        self.__data_capture_p = th.Thread(target=self.__trajectory_handle.capture_data, args=(self.__basecyclic,))
        self.__data_capture_p.start()

        # home the robot arm
        print(f'Moving to home position')
        self.home()
        self.__control_mode = mode

    def move(self, data: Dict, feedback: bool = False):
        """ Move to a certain cartesian/angles pos or with a specific twist based
        on the control_mode.

        Args:
            data (Dict): Set of joint/cartesian pos or twists.
            feedback (bool, optional): Upon activation, feedback loggers in other threads
                start recording. Defaults to False.
        """
        if feedback:
            self.__trajectory_handle.capture = True

        if self.__control_mode == ControlModes.END_EFFECTOR_POSE:
            print(f'Moving to ({data["linear_x"]}, {data["linear_y"]}, {data["linear_z"]})')
            endeffector_pose_command(self.__base, endeffector_pose_dict=data)

        elif self.__control_mode == ControlModes.END_EFFECTOR_TWIST:
            print(f'Moving with ({data["linear_x"]}, {data["linear_y"]}, {data["linear_z"]})')
            endeffector_twist_command(self.__base, duration=data["duartion"],
                                      endeffector_twists_dict=data)

        elif self.__control_mode == ControlModes.JOINT_POSITION:
            joints_position_command(self.__base, joint_positions_dict=data)

        elif self.__control_mode == ControlModes.JOINT_VELOCITY:
            joints_velocity_command(self.__base, duration=data["duartion"],
                                    joint_velocity_dict=data)

    def home(self):
        # activate single level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.__base.SetServoingMode(base_servo_mode)

        # Move arm to ready position
        print("Moving the arm to a safe position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.__base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        if action_handle == None:
            print("Can't reach safe position. Exiting")

        e = threading.Event()
        notification_handle = self.__base.OnNotificationActionTopic(
            partial(check, e=e),
            Base_pb2.NotificationOptions()
        )

        self.__base.ExecuteActionFromReference(action_handle)

        # leave time to action to complete
        finished = e.wait(15000)
        self.__base.Unsubscribe(notification_handle)

        return finished

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__trajectory_handle.capture = False
        self.__data_capture_p.join()
        self.__trajectory_handle.save_data()
        self._terminate_connection()

    def _terminate_connection(self):
        # terminate everything
        self.__base.Stop()

        router_options = RouterClientSendOptions()
        router_options.timeout_ms = 1000

        self.__session_manager.CloseSession(router_options)
        self.__transport.disconnect()

    def get_trajectory(self):
        return self.__trajectory_handle

if __name__ == '__main__':
    with KinovaDSExperiments() as kde:
        kde.move(kde.get_trajectory().start, feedback=True)
        pass
