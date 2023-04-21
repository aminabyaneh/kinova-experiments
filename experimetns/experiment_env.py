#! /usr/bin/env python3

import math
import os, sys
from typing import List, Tuple
import numpy as np
import threading as th
import pandas as pd
import pickle

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
        # TODO: Could potentially write a parser for coordinates.
        self.start = {'linear_x': 0.006,
                      'linear_y': -0.414,
                      'linear_z': 0.137,
                      'angular_x': -2.338,
                      'angular_y': 177.674,
                      'angular_z': 177.56}

        self.goal = {'linear_x': 0.306,
                     'linear_y': 0.327,
                     'linear_z': 0.215,
                     'angular_x': 177.643,
                     'angular_y': 2.36,
                     'angular_z': 144.643}

        self.pos_dataset = pd.DataFrame(columns=['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])
        self.vel_dataset = pd.DataFrame(columns=['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])

        self.capture: bool = False

    def capture_data(self, basecyclic: BaseCyclicClient, delta_t: float = 0.1):
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
            vel_dict = endeffector_twist_feedback(basecyclic)
            pos_dict = endeffector_pose_feedback(basecyclic)

            self.pos_dataset = pd.concat([self.pos_dataset, pd.DataFrame(pos_dict, index=[n_samples])])
            self.vel_dataset = pd.concat([self.vel_dataset, pd.DataFrame(vel_dict, index=[n_samples])])

            time.sleep(delta_t)
            n_samples += 1

        print(f'Terminating demonstration data logger with {n_samples} samples')

    def clear_data(self):
        """ Clear datasets to recapture another sequence.
        """

        self.pos_dataset = pd.DataFrame(columns=['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])
        self.vel_dataset = pd.DataFrame(columns=['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])

    def save_data(self, dir: str = os.getcwd()):
        """Save the captured data from demonstrations.

        Args:
            dir (str, optional): Path to dems folder contatining the pos and vels. Defaults to os.getcwd().
        """

        save_dir = os.path.join(dir, 'dems')
        os.makedirs(save_dir, exist_ok=True)
        self.pos_dataset.to_csv(path_or_buf=os.path.join(save_dir, 'pos.csv'))
        self.vel_dataset.to_csv(path_or_buf=os.path.join(save_dir, 'vel.csv'))

    def load_data(self, dir: str = os.getcwd()):
        """Load a previously stored trajectory.

        Args:
            dir (str, optional): Directory of dems folder contatining pos and velocity data. Defaults to os.getcwd().
        """

        load_dir = os.path.join(dir, 'dems')
        self.pos_dataset = pd.read_csv(os.path.join(load_dir, 'pos.csv'), index_col=0)
        self.vel_dataset = pd.read_csv(os.path.join(load_dir, 'vel.csv'), index_col=0)


class KinovaDSExperiments:
    def __init__(self, mode=ControlModes.END_EFFECTOR_POSE, device_ip: str = '192.168.1.10',
                 device_port: int = 10000, username: str = 'admin', password: str = 'admin',
                 session_inactivity_timeout: int = 60000, connection_inactivity_timeout: int = 20000,
                 capture_mode: bool = False, home: bool = True):

        # build the transport layer
        self.__transport = TCPTransport() if device_port == DeviceConnection.TCP_PORT else UDPTransport()
        self.__router = RouterClient(self.__transport, RouterClient.basicErrorCallback)
        self.__transport.connect(device_ip, device_port)
        self.__trajectory_handle = RealWorldTrajectory()
        self.__capture_mode = capture_mode


        # ds planning policy
        self.__ds_planner = None

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
        if self.__capture_mode:
            self.__data_capture_p = th.Thread(target=self.__trajectory_handle.capture_data, args=(self.__basecyclic,))
            self.__data_capture_p.start()

        # home the robot arm
        if home:
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
        if feedback and self.__capture_mode:
            self.__trajectory_handle.capture = True

        if self.__control_mode == ControlModes.END_EFFECTOR_POSE:
            endeffector_pose_command(self.__base, endeffector_pose_dict=data)

        elif self.__control_mode == ControlModes.END_EFFECTOR_TWIST:
            endeffector_twist_command(self.__base, duration=data["dt"],
                                      endeffector_twists_dict=data)

        elif self.__control_mode == ControlModes.JOINT_POSITION:
            joints_position_command(self.__base, joint_positions_dict=data)

        elif self.__control_mode == ControlModes.JOINT_VELOCITY:
            joints_velocity_command(self.__base, duration=data["dt"],
                                    joint_velocity_dict=data)

    def grip(self, press: float = 0.7):
        """ Close the gripper.
        """
        change_gripper(self.__base, press)

    def release(self):
        """ Release the gripper.
        """
        change_gripper(self.__base, 0.0)

    def pause(self, secs=2):
        """ Pause for a determined time. Only a wrapper for sleep at this point.

        Args:
            secs (int, optional): Seconds to pause. Defaults to 2.
        """
        time.sleep(secs)

    def home(self):
        """ Move to a predefined home position.
        """
        # activate single level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.__base.SetServoingMode(base_servo_mode)

        # move arm to ready position
        print("Moving the arm to a safe position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.__base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        e = threading.Event()
        notification_handle = self.__base.OnNotificationActionTopic(
            partial(check, e=e),
            Base_pb2.NotificationOptions()
        )

        self.__base.ExecuteActionFromReference(action_handle)

        # leave time to action to complete
        e.wait(15000)
        self.__base.Unsubscribe(notification_handle)

    def get_endeffector_feedback(self):
        """Return a dict with joints pose and twists feedback.

        Returns:
            Dict: full feedback
        """
        return {"pose": endeffector_pose_feedback(self.__basecyclic), "twist": endeffector_twist_feedback(self.__basecyclic)}

    def get_joints_feedback(self):
        """Return a dict with joints pose and twists feedback.

        Returns:
            Dict: full feedback
        """
        return {"position": joints_position_feedback(self.__basecyclic), "velocity": joints_velocity_feedback(self.__basecyclic)}

    def inverse_kinematics(self, pose):
        """Calculate the inverse kinematics.

        Args:
            pose (dict): End-effector Position

        Returns:
            dict: Joint positions optimized by inverse kinematics.
        """

        init_angles_guess = self.get_joints_feedback()["position"]
        joint_angles = inverse_kinematics(self.__base, pose, init_angles_guess)
        return {joint_id: joint_angles.joint_angles[joint_id].value for joint_id in range(6)}

    def execute_trajectory(self, trajectory, is_joint_space: bool = False):
        """Execute a trajectory.

        Args:
            trajectory (List): A list of all the waypoints in the trajectory.
            is_joint_space (bool, optional): False means it's a cartesian trajectory. Defaults to False.
        """

        if is_joint_space:
            execute_jointspace_trajectory(self.__base, trajectory)
        else:
            execute_taskspace_trajectory(self.__base, trajectory)

    def get_trajectory(self):
        """Get the trajectory handler.

        Returns:
            RealWorldTrajectory: the handler
        """

        return self.__trajectory_handle

    def set_control_mode(self, mode: ControlModes):
        """ Set the arm's control mode.

        Args:
            mode (ControlModes): _description_
        """

        assert mode in ControlModes, "Invalid control mode!"
        self.__control_mode = mode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__capture_mode:
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


def reproduce_trajectory():
    with KinovaDSExperiments() as kde:
        traj = kde.get_trajectory()
        traj.load_data()

        dt = 0.1
        kde.set_control_mode(ControlModes.END_EFFECTOR_TWIST)

        for index, row in traj.vel_dataset.iterrows():
            twists_dict = dict(row)
            twists_dict["dt"] = dt
            kde.move(twists_dict, feedback=False)
            print(f'Sending {index} twist command')
        pass


def baseline_sine_motion():
    start_point = {'linear_x': 0.117,
                    'linear_y': 0.00,
                    'linear_z': 0.255,
                    'angular_x': 1.393,
                    'angular_y': 178.674,
                    'angular_z': 96.029}

    with KinovaDSExperiments() as kde:
        kde.pause(secs=5)
        kde.grip(press=0.8)

        kde.set_control_mode(ControlModes.JOINT_POSITION)

        joint_position_start = kde.inverse_kinematics(start_point)
        print(f'Moving to ({start_point["linear_x"]:.3f}, {start_point["linear_y"]:.3f}, {start_point["linear_z"]:.3f})')
        kde.move(joint_position_start)

        pose_fb = kde.get_endeffector_feedback()["pose"]
        xb, yb, zb = pose_fb['linear_x'], pose_fb['linear_y'], pose_fb['linear_z']
        tx, ty, tz = pose_fb['angular_x'], pose_fb['angular_y'], pose_fb['angular_z']

        # generate a trajectory
        task_space_trajectory: List[Tuple] = []
        x = xb
        while x < 0.435:
            # calculate new end-effector positions
            x_n = x + 0.005
            y_n = yb + 0.15 * math.sin(50 * (x_n - xb))

            # move to calculated position
            print(f'Adding ({x_n:.3f}, {y_n:.3f}, {zb:.3f})')
            target = {'linear_x': x_n, 'linear_y': y_n, 'linear_z': zb, 'angular_x': tx, 'angular_y': ty, 'angular_z': tz}
            task_space_trajectory.append(target)

            # feedback simulation
            x = x_n

        kde.execute_trajectory(task_space_trajectory)

def baseline_w_motion():
    start_point = {'linear_x': 0.100,
                    'linear_y': 0.00,
                    'linear_z': 0.255,
                    'angular_x': 1.393,
                    'angular_y': 178.674,
                    'angular_z': 96.029}

    with KinovaDSExperiments() as kde:
        kde.pause(secs=5)
        kde.grip(press=0.8)

        kde.set_control_mode(ControlModes.JOINT_POSITION)

        joint_position_start = kde.inverse_kinematics(start_point)
        print(f'Moving to ({start_point["linear_x"]:.3f}, {start_point["linear_y"]:.3f}, {start_point["linear_z"]:.3f})')
        kde.move(joint_position_start)

        pose_fb = kde.get_endeffector_feedback()["pose"]
        xb, yb, zb = pose_fb['linear_x'], pose_fb['linear_y'], pose_fb['linear_z']
        tx, ty, tz = pose_fb['angular_x'], pose_fb['angular_y'], pose_fb['angular_z']

        # generate a trajectory
        task_space_trajectory: List[Tuple] = []
        x = xb
        for _ in range(56):
            # calculate new end-effector positions
            x_n = x + 0.005
            y_n = yb + 0.1 * math.sqrt(abs(3 - ((20*(x_n - 0.24)) ** 2)))

            # move to calculated position
            print(f'Adding ({x_n:.3f}, {y_n:.3f}, {zb:.3f})')
            target = {'linear_x': x_n, 'linear_y': y_n, 'linear_z': zb, 'angular_x': tx, 'angular_y': ty, 'angular_z': tz}
            task_space_trajectory.append(target)

            # feedback simulation
            x = x_n

        target = {'linear_x': 0.381, 'linear_y': 0.25 * math.sqrt(2 - ((10*(0.381 - 0.24)) ** 2)), 'linear_z': zb, 'angular_x': tx, 'angular_y': ty, 'angular_z': tz}
        kde.execute_trajectory(task_space_trajectory)


def baseline_c_motion():
    start_point = {'linear_x': 0.100,
                    'linear_y': 0.00,
                    'linear_z': 0.255,
                    'angular_x': 1.393,
                    'angular_y': 178.674,
                    'angular_z': 96.029}

    with KinovaDSExperiments() as kde:
        kde.pause(secs=5)
        kde.grip(press=0.8)

        kde.set_control_mode(ControlModes.JOINT_POSITION)

        joint_position_start = kde.inverse_kinematics(start_point)
        print(f'Moving to ({start_point["linear_x"]:.3f}, {start_point["linear_y"]:.3f}, {start_point["linear_z"]:.3f})')
        kde.move(joint_position_start)

        pose_fb = kde.get_endeffector_feedback()["pose"]
        xb, yb, zb = pose_fb['linear_x'], pose_fb['linear_y'], pose_fb['linear_z']
        tx, ty, tz = pose_fb['angular_x'], pose_fb['angular_y'], pose_fb['angular_z']

        # generate a trajectory
        task_space_trajectory: List[Tuple] = []
        x = xb
        for _ in range(56):
            # calculate new end-effector positions
            x_n = x + 0.005
            y_n = yb + 0.25 * math.sqrt(2 - ((10*(x_n - 0.24)) ** 2))

            # move to calculated position
            print(f'Adding ({x_n:.3f}, {y_n:.3f}, {zb:.3f})')
            target = {'linear_x': x_n, 'linear_y': y_n, 'linear_z': zb, 'angular_x': tx, 'angular_y': ty, 'angular_z': tz}
            task_space_trajectory.append(target)

            # feedback simulation
            x = x_n

        target = {'linear_x': 0.381, 'linear_y': 0.25 * math.sqrt(2 - ((10*(0.381 - 0.24)) ** 2)), 'linear_z': zb, 'angular_x': tx, 'angular_y': ty, 'angular_z': tz}
        kde.execute_trajectory(task_space_trajectory)


def sample_pick_and_place():
    with KinovaDSExperiments() as kde:
        traj = kde.get_trajectory()
        kde.set_control_mode(ControlModes.END_EFFECTOR_POSE)

        # release the gripper and move to the starting point
        kde.release()
        kde.move(traj.start)
        kde.pause()

        # grip and move to the goal
        kde.grip()
        kde.move(traj.goal)
        kde.pause()

        # release
        kde.release()
        kde.home()

if __name__ == '__main__':
    baseline_w_motion()