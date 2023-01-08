#!/usr/bin/env python

from typing import List

import rospy
import math
import rosparam
import argparse

from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import SetBool
from tf.transformations import euler_from_quaternion


# avoide redundancy in state logs
show_state_log: bool = True


class States:
    IDLE = "IDLE"
    GOAL = "GOAL"
    WALL = "WALL"
    GOAL_ADJUST_HEADING = "ADJUST_HEADING"
    GOAL_GO_STRAIGHT = "GO_STRAIGHT"
    GOAL_ACHIEVED = "ACHIEVED"
    WALL_SEARCH = "SEARCH"
    WALL_TURN = "TURN"
    WALL_FOLLOW = "FOLLOW"


def min_d(array_like: List, tolerance: int = 3):
    sorted_array = array_like.sort()
    return sum(sorted_array[:3]) / 3


class BaseController:
    def __init__(self, map: str = "cafe"):

        # general configs
        self.__is_active = True
        self.__camera_topic_name = "/scan"
        self.__activate_service_name = "/base_controller/start"
        self.__config_file_address = f'../config/{map}.yaml'
        self.__odom_topic_name = '/gazebo/model_states'
        self.__configs = rosparam.load_file(self.__config_file_address,
                                            default_namespace="base_controller")[0][0]
        rospy.loginfo(f'Configuration parsed: \n{self.__configs}')

        # some configs might not be available in some of the files
        try:
            self.__min_dist_to_obstacles = self.__configs["min_distance_to_obstacles"]
        except KeyError:
            self.__min_dist_to_obstacles = 1.0
            rospy.logwarn('No minimum distance to obstacles is defined.')

        try:
            self.__speed_lim = self.__configs["limit"]["speed"]
            self.__omega_lim = self.__configs["limit"]["omega"]
        except KeyError:
            self.__speed_lim = 1.0
            self.__omega_lim = 1.0
            rospy.logwarn('No speed limit is defined.')

        if map == "agriculture" or map == "orchard" or map == "inspection":
            self.__remove_noise = True
        else:
            self.__remove_noise = False
        self.__strict_regions = True

        # motion configurations
        self.__yaw_precision = 0.05
        self.__yaw_adj_speed = 0.5 * self.__omega_lim

        self.__position_precision = 0.5
        self.__dist_goal_tolerance = 0.1
        self.__obstacle_dist_tolerance = 0.05
        self.__dist_to_start_goal_line_precision = 0.05
        self.wall_timeout = 2

        self.__turn_left_speed_low = 0.5 * self.__omega_lim
        self.__turn_left_speed_fast = 0.8 * self.__omega_lim
        self.__forward_speed_slow = 0.4 * self.__speed_lim
        self.__forward_speed_fast = 0.8 * self.__speed_lim


        # class variables
        self.__region = None
        self.__new_camera_data = True
        self.__wall_observed = 0
        self.__new_camera_data = True
        self.__camera_fault_tolerance = 10
        self.__noise_value = 0.6

        self.__goal_state = States.GOAL_ADJUST_HEADING
        self.__wall_state = States.WALL_SEARCH

        self.state = States.GOAL

        self.__current_position = Point()
        self.__current_orientation = Point()
        self.__current_linearvel = Twist.angular
        self.__current_angularvel = Twist.linear

        self.__initial_position = Point()
        self.__final_position = Point()
        self.__set_init_final_positions()

        self.__hitpoint = Point()
        self.__leave_position = Point()
        self.__dist_to_goal_from_hitpoint = None
        self.__dist_to_goal_from_leave_point = None

        # publisher subscribers
        self.__velocity_publisher = rospy.Publisher(self.__configs["cmd_topic_name"],
                                                    Twist, queue_size=10)

        self.__camera_subscriber = rospy.Subscriber(self.__camera_topic_name, LaserScan,
                                                    self.__camera_callback)

        self.__activate_service = rospy.Service(self.__activate_service_name, SetBool,
                                                self.__activate_callback)

        self.__odom_subscriber = rospy.Subscriber(self.__odom_topic_name , ModelStates,
                                                  self.__odom_callback)

    def __dist_to_line(self, point):
        # beware of the none inputs
        assert point.x is not None, "x cannot be NoneType!"
        assert point.y is not None, "y cannot be NoneType!"

        # use geometry to calculate distance
        num = self.__initial_position.y - self.__final_position.y
        denum = self.__initial_position.x - self.__final_position.x

        denum = denum if denum != 0.0 else denum + 0.01
        m = num / denum

        numerator = math.fabs(point.y - self.__initial_position.y - m * point.x + m * self.__initial_position.x)
        denominator = math.sqrt(1 + math.pow(m, 2))
        dist = numerator / denominator

        return dist

    def __set_init_final_positions(self):
        try:
            self.__initial_position.x = self.__configs["initial"]["x"]
            self.__initial_position.y = self.__configs["initial"]["y"]
            self.__initial_position.z = self.__configs["initial"]["phi"]
            rospy.loginfo(f'Initial position is ({self.__initial_position.x:.2f}, '
                          f'{self.__initial_position.y:.2f}).')
        except KeyError:
            rospy.logwarn('No initial position in the configuration.')

        try:
            self.__final_position.x = self.__configs["target"]["x"]
            self.__final_position.y = self.__configs["target"]["y"]
            self.__final_position.z = self.__configs["target"]["phi"]
            rospy.loginfo(f'Target position is ({self.__final_position.x:.2f}, '
                          f'{self.__final_position.y:.2f}).')
        except KeyError:
            rospy.logwarn('No initial position in the configuration.')

    def __odom_callback(self, msg):
        msg_id = msg.name.index("husky")

        husky_pose = msg.pose[msg_id]
        husky_twist = msg.twist[msg_id]

        rospy.logdebug(f'Navigation data is fetched ({husky_pose.position.x:.2f}, {husky_pose.position.y:.2f}, {husky_pose.orientation.z:.2f}).')
        self.__current_position = husky_pose.position

        x = husky_pose.orientation.x
        y = husky_pose.orientation.y
        z = husky_pose.orientation.z
        w = husky_pose.orientation.w
        quaternion = (x, y, z, w)

        euler = euler_from_quaternion(quaternion)
        self.__current_orientation = Point()
        self.__current_orientation.x = euler[0]
        self.__current_orientation.y = euler[1]
        self.__current_orientation.z = euler[2]

        self.__current_linearvel = husky_twist.linear
        self.__current_angularvel = husky_twist.angular

    def __activate_callback(self, req):
        rospy.logwarn(f'Setting the activation parameter to {req.data}.')
        self.__is_active = req.data

    def __camera_callback(self, msg):
        if self.__remove_noise:
            noise_free_msg = [read if read > self.__noise_value else 10.0 for read in msg.ranges]
        else:
            noise_free_msg = msg.ranges

        self.__region = {"eleft": min(min(noise_free_msg[0:143]), 10),
                         "left":  min(min(noise_free_msg[144:287]), 10),
                         "center": min(min(noise_free_msg[288:431]), 10),
                         "right":   min(min(noise_free_msg[432:575]), 10),
                         "eright":  min(min(noise_free_msg[576:719]), 10)}

        if self.__strict_regions:
            self.__region["left"] = min(self.__region["left"], self.__region["eleft"])
            self.__region["right"] = min(self.__region["right"], self.__region["eright"])

        self.__new_camera_data = True
        rospy.loginfo(f'Camera message received ({self.__region["left"]:.2f},'
                      f' {self.__region["center"]:.2f}, {self.__region["right"]:.2f}).')

    def __move_robot(self, x, y, d):
        rospy.loginfo(f'Moving robot with ({x:.2f}, {y:.2f}, {d:.2f}).')
        twists = Twist()
        twists.linear.x = x
        twists.linear.y = y
        twists.linear.z = 0

        twists.angular.x = 0
        twists.angular.y = 0
        twists.angular.z = d

        # publish the velocities
        self.__velocity_publisher.publish(twists)

    def __calculate_dist(self, point_0, point_1):
        x_dist = pow(point_0.x - point_1.x, 2)
        y_dist = pow(point_0.y - point_1.y, 2)
        return math.sqrt(x_dist + y_dist)

    def is_active(self):
        return self.__is_active

    def go_to_goal(self):
        d_threshold = self.__min_dist_to_obstacles - self.__obstacle_dist_tolerance

        # return if camera data is not there yet
        if self.__region is None: return

        # check if a wall is on the way
        if (self.__region["left"] < d_threshold or
            self.__region["center"] < d_threshold or
            self.__region["right"] < d_threshold):

            if not self.__new_camera_data: return

            self.__wall_observed += 1
            self.__new_camera_data = False

            if self.__wall_observed <= self.__camera_fault_tolerance:
                rospy.logwarn(f'Fault tolerance ({self.__wall_observed}/{self.__camera_fault_tolerance}) discarded ({self.__region["left"]:.2f},'
                      f' {self.__region["center"]:.2f}, {self.__region["right"]:.2f}).')
                return

            rospy.logdebug(f'Wall spoted on the way with {d_threshold:.2f}.')

            self.state = States.WALL

            # preserve the hit point and distance from it
            self.__hitpoint.x = self.__current_position.x
            self.__hitpoint.y = self.__current_position.y
            self.__dist_to_goal_from_hitpoint = \
                self.__calculate_dist(self.__hitpoint, self.__final_position)
            rospy.logwarn(f'Hit-point recorded at ({self.__hitpoint.x:.2f}, {self.__hitpoint.y:.2f}).')

            self.__move_robot(0.0, 0.0, self.__turn_left_speed_fast)
            return

        else:
            self.__wall_observed = 0

        rospy.loginfo(f'Current goal state: {self.__goal_state}')
        if self.__goal_state == States.GOAL_ADJUST_HEADING:
            desired_yaw = math.atan2(
                    self.__final_position.y - self.__current_position.y,
                    self.__final_position.x - self.__current_position.x)

            # calculate the heading and turn if necessary
            yaw_error = desired_yaw - self.__current_orientation.z
            rospy.loginfo(f'Desired yaw is {desired_yaw:.2f}, current yaw is {self.__current_orientation.z:.2f}.')
            rospy.logdebug(f'Yaw error is {yaw_error}.')

            if math.fabs(yaw_error) > self.__yaw_precision:
                if yaw_error > 0:
                    self.__move_robot(0.0, 0.0, self.__yaw_adj_speed)
                else:
                    self.__move_robot(0.0, 0.0, -self.__yaw_adj_speed)

            # change the state if the heading is good enough
            else:
                rospy.logwarn('Heading is adjusted, moving forward.')
                self.__goal_state = States.GOAL_GO_STRAIGHT
                self.__move_robot(0.0, 0.0, 0.0)

        if self.__goal_state == States.GOAL_GO_STRAIGHT:
            position_error = self.__calculate_dist(self.__current_position,
                                                   self.__final_position)

            # check if the goal is reached yet
            if position_error > self.__position_precision:
                self.__move_robot(self.__forward_speed_fast, 0.0, 0.0)

                # go back to the adjustment if heading is wrong
                desired_yaw = math.atan2(
                    self.__final_position.y - self.__current_position.y,
                    self.__final_position.x - self.__current_position.x)
                yaw_error = desired_yaw - self.__current_orientation.z

                if math.fabs(yaw_error) > self.__yaw_precision:
                    self.__goal_state = States.GOAL_ADJUST_HEADING
            else:
                self.__goal_state = States.GOAL_ACHIEVED
                self.__move_robot(0.0, 0.0, 0.0)

        if self.__goal_state == States.GOAL_ACHIEVED:
            rospy.loginfo(f'Goal is reached now at ({self.__current_position.x}, {self.__current_position.y}).')
            self.state = States.IDLE
            self.__move_robot(0.0, 0.0, 0.0)
            pass

    def follow_wall(self):
        rospy.loginfo(f'Current wall state: {self.__wall_state}')
        num = self.__initial_position.y - self.__final_position.y
        denum = self.__initial_position.x - self.__final_position.x

        denum = denum if denum != 0.0 else denum + 0.01
        start_goal_line_slope = num / denum

        start_goal_line_intercept = (self.__final_position.y - start_goal_line_slope * self.__final_position.x)
        rospy.loginfo(f'Start-goal line slope is {start_goal_line_slope:.2f} '
                      f'and intercept is {start_goal_line_intercept:.2f}.')

        # calculate distance to start-goal line
        dist_to_start_goal_line = self.__dist_to_line(self.__current_position)
        rospy.loginfo(f'Distance to start-goal line is {dist_to_start_goal_line:.3f}.')

        # determine if we need to leave the wall and change the mode
        if dist_to_start_goal_line < self.__dist_to_start_goal_line_precision:
            self.__leave_position = self.__current_position
            self.__dist_to_goal_from_leave_point = self.__calculate_dist(self.__leave_position,
                                                                         self.__final_position)

            if self.__dist_to_goal_from_leave_point < self.__dist_to_goal_from_hitpoint - self.__dist_goal_tolerance:
                rospy.loginfo(f'Distance to goal now {self.__dist_to_goal_from_leave_point:.2f} and from hit {self.__dist_to_goal_from_hitpoint:.2f}.')
                self.__goal_state = States.GOAL_ADJUST_HEADING
                self.state = States.GOAL
                return

        # if not reached to the start-goal line, follow the wall
        d = self.__min_dist_to_obstacles + self.__obstacle_dist_tolerance

        if self.__region["left"] > d and self.__region["center"] > d and self.__region["right"] > d:
            self.__wall_state = States.WALL_SEARCH
            self.__move_robot(self.__forward_speed_slow, 0.0, -self.__turn_left_speed_fast)

        elif self.__region["left"] > d and self.__region["center"] < d and self.__region["right"] > d:
            self.__wall_state = States.WALL_TURN
            self.__move_robot(0.0, 0.0, self.__turn_left_speed_low)

        elif (self.__region["left"] > d and self.__region["center"] > d and self.__region["right"] < d):
            if (self.__region["right"] <= 0.8 * d):
                rospy.logwarn(f'Too close to the wall, turning immediately.')
                self.__wall_state = States.WALL_TURN
                self.__move_robot(self.__forward_speed_slow, 0.0, self.__turn_left_speed_fast)
            else:
                self.__wall_state = States.WALL_FOLLOW
                self.__move_robot(self.__forward_speed_fast, 0.0, 0.0)

        elif self.__region["left"] < d and self.__region["center"] > d and self.__region["right"] > d:
            self.__wall_state = States.WALL_SEARCH
            self.__move_robot(0.0, 0.0, self.__turn_left_speed_fast)

        elif self.__region["left"] > d and self.__region["center"] < d and self.__region["right"] < d:
            self.__wall_state = States.WALL_TURN
            self.__move_robot(0.0, 0.0, self.__turn_left_speed_low)

        elif self.__region["left"] < d and self.__region["center"] < d and self.__region["right"] > d:
            self.__wall_state = States.WALL_TURN
            self.__move_robot(0.0, 0.0, self.__turn_left_speed_low)

        elif self.__region["left"] < d and self.__region["center"] < d and self.__region["right"] < d:
            self.__wall_state = States.WALL_TURN
            self.__move_robot(0.0, 0.0, self.__turn_left_speed_low)

        elif self.__region["left"] < d and self.__region["center"] > d and self.__region["right"] < d:
            self.__wall_state = States.WALL_SEARCH
            self.__move_robot(0.0, 0.0, -self.__turn_left_speed_low)

        else:
            rospy.logwarn(f'No category found for wall-follow.')
            pass

    def is_wondering(self):
        return self.__wall_state == States.WALL_SEARCH

def main(selected_map: str):
    rospy.init_node("base_controller", anonymous=True)
    rospy.logwarn(f'Selected simulation map is {selected_map}.')
    bc_module = BaseController(map=selected_map)

    rate = rospy.Rate(200)
    time, wall_search_time = rospy.rostime.Duration(secs=0), rospy.rostime.Duration(secs=0)
    while not rospy.is_shutdown():
        if not bc_module.is_active():
            rospy.loginfo('Base controller is not activated.')
            rate.sleep()
            continue

        if bc_module.state == States.GOAL:
            rospy.loginfo('Current state: Toward Goal')
            bc_module.go_to_goal()
            time = rospy.rostime.get_rostime()

        elif bc_module.state == States.WALL:
            if bc_module.is_wondering():
                wall_search_time = rospy.rostime.get_rostime() - time
            else:
                wall_search_time = rospy.rostime.Duration(secs=0)
                time = rospy.rostime.get_rostime()

            if wall_search_time.to_sec() > bc_module.wall_timeout:
                rospy.logwarn(f'Time out for wall search, check moving toward goal.')
                bc_module.state = States.GOAL
                continue

            rospy.loginfo(f'Current state: Wall Follow ({wall_search_time.to_sec():.2f})')
            bc_module.follow_wall()

        elif bc_module.state == States.IDLE:
            rospy.logwarn(f'Goal is finally reached, terminating the controller.')
            break
        rate.sleep()

    rospy.signal_shutdown("Controller served its purpose.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--selected-map",
                        help = "Pick a map from the following list:"
                        "\n1. agriculture\n2.cafe\n3.empty\n4.inspection"
                        "\n5.mud\n6.office\n7.orchard", default="cafe")

    args = parser.parse_args()

    try:
        main(args.selected_map)
    except rospy.exceptions.ROSInterruptException:
        rospy.logerr(f'Shutting down peacefully due to a user interrupt.')