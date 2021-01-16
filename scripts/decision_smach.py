#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import sys, time
# ros_path = '/opt/ros/noetic/lib/python2.7/dist-packages'
# if ros_path in sys.path:
#     sys.path.remove(ros_path)
# from numba import jit
# sys.path.append(ros_path)
from numba import jit

import rospy
import smach
import smach_ros
import copy
import numpy as np
import math
import threading
from multiprocessing.pool import ThreadPool
from local_messages.msg import FilteredObstacle
from local_messages.msg import Decision
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Vector3
from local_messages.msg import Road
from local_messages.msg import Lane
from local_messages.msg import GlobalPose
from local_messages.msg import Obstacles
from local_messages.msg import Obstacle
from local_messages.msg import Lights
from local_messages.msg import Light
from local_messages.msg import Signs
from local_messages.msg import Sign
from local_messages.msg import Things
from local_messages.msg import Thing
from local_messages.msg import ControlFeedback
from local_messages.msg import Missions
from shapely.geometry import Point, Polygon

from local_messages.srv import PlanningFeedback, PlanningFeedbackRequest, PlanningFeedbackResponse
from local_messages.srv import ReGlobalPlanning, ReGlobalPlanningRequest, ReGlobalPlanningResponse
from local_messages.srv import MissionFinished, MissionFinishedRequest, MissionFinishedResponse

# import all the msg and srv files


# stop line type from map
NO_STOP = 0 # default value
SLOW_GIVE_WAY = 1
STOP_GIVE_WAY = 2
OBSERVE_TRAFFIC_LIGHT = 3
LANE_END = 4
DESTINATION = 5

# For a non-zero value, the driver may need to look for traffic lights, and related turnning regulations.
NO_TURN = 0
STRAIGHT = 1
LEFT = 2
RIGHT = 3
U_TURN = 4


# The type of the light.
ARROW_UP = 1
ARROW_LEFT = 2
ARROW_RIGHT = 3
ARROW_DOWN = 4
CIRCLE = 5

# The color of the light.
RED = 1
YELLOW = 2
GREEN = 3

# velocity defined by m/s, while the speed upper limit on map is defined by km/h

EPS = 0.001

MIN_TURNING_RADIUS = 4.5
VEHICLE_WIDTH = 1.86
VEHICLE_LENGTH = 5
LANE_CHANGE_BASE_LENGTH = 12
OBSERVE_RANGE = 60
LANE_WIDTH_BASE = 3.7
STOP_BASE_LENGTH = 2.5
# 障碍物的感知距离（用于对车道可行驶距离做限制，期望路径的距离做限制）

# Planning feedback
NONE = 0
RE_CHOOSE_PATH = 1
RE_CHOOSE_PARKING_SPOT = 2
PARKING_DONE = 3
EXIT_PARKING_DONE = 4

# 大决策：当前自车的行为
# possible values
# NONE = 0
REF_PATH_FOLLOW = 1
EMERGENCY_BRAKE = 2
PARK = 3
EXIT_PARK = 4
# UTURN = 5

# 动静态障碍物
# Is the obstacle static or dynamic?
STATIC = 0
DYNAMIC = 1

# 小决策
# The reaction towards the obstacle
# NONE = 0
OVERTAKE = 1
GIVE_WAY = 2
AVOID_COLLISION = 3

PREDICTION_DURATION = 5
PREDICTION_PERIOD = 0.1

DECISION_PERIOD = 0.5

SPEED_UPPER_LIMIT_DEFAULT = 30
SPEED_LOWER_LIMIT_DEFAULT = 0
SPEED_UPPER_LIMIT_MERGE = 1.5
SPEED_UPPER_LIMIT_INTERSECTION = 3
SPEED_UPPER_LIMIT_APPROACH = 5

COMFORT_DEC = 3

TIME_ACC = 1
# is computed as the time for accelerating from zero up to speed in the destination lane using the same conservative acceleration.
TIME_DELAY = 1
# is estimated as the maximum system delay.
TIME_SPACE = 1
# is defined as the minimum required temporal spacing between vehicles, where 1 s approximates a vehicle length per 10 mph.
TIME_LANE_CHANGE_CONSIDERATION = 1.5
# TIME_LANE_CHANGE_CALM_DOWN = 1
TIME_LANE_CHANGE_DEADLINE = 15

MIN_GAP_DISTANCE = 7.5  # One car length
MIN_SAFE_DISTANCE = 0.2
MEDIUM_SAFE_DISTANCE = 0.5

PARALLEL_SLOT = 1
VERTICAL_SLOT = 2

class MissionAhead:
    def __init__(self):
        self.missionType = None
        self.missionLocationX = 0
        self.missionLocationY = 0
        self.missionLaneIds = []
        self.missionIndex = 0
        self.missionThingId = 0

mission_ahead = MissionAhead()
global_pose_data = None
lane_list = {}
obstacles_list = {}
signs_data = {}
lights_list = {}
parking_slots_list = {}
parking_area_list = {}
planning_feedback = 0

road_updated_flag = 1
pose_updated_flag = 1
lights_updated_flag = 1
signs_updated_flag = 1
map_things_updated_flag = 1
parking_slot_updated_flag = 1
obstacle_updated_flag = 1
user_data_copied_flag = 1

mission_completed = 0

target_parking_area = []
target_parking_slot = []
target_parking_slot_projection = []
parking_lane_id = 0

# ready_to_go = 0

history_lane_ids = []
last_target_lane_id_ugv = -1
last_chosen_lane_id_ugv = -1
wait_duration = 0
lane_change_consideration_start_time = 0
# lane_change_calm_down_start_time = 0
action_done = False
last_stop_x = -1
scenario_error_handle = REF_PATH_FOLLOW
bb = 0

class Pose:
    def __init__(self):
        self.mapX = 0
        self.mapY = 0
        self.mapHeading = 0


class CurrentLaneInfo:
    """
    update lane information, triggered when the global pose is updated.
    记录关于当前车道的动态信息
    """
    def __init__(self):
        # id 为正数说明又车道
        self.cur_lane_id = -1
        self.left_lane_id = -1
        self.right_lane_id = -1
        self.projection_index = -1
        self.projection_x = -1
        self.projection_y = -1
        self.lateral_distance = -1
        self.dir_diff_signed = -1
        self.before_length = -1
        self.after_length = -1
        self.next_stop_type = -1
        self.next_stop_x = -1
        self.dist_to_next_stop = 0
        self.dist_to_next_road = 0
        self.cur_turn_type = -1
        self.next_turn_type = -1
        self.can_change_left = 0
        self.can_change_right = 0
        self.speed_upper_limit = 0
        self.speed_lower_limit = 0
        self.cur_priority = -1
        self.left_priority = []
        self.right_priority = []


class DrivableLanes:
    def __init__(self):
        self.id = 0
        self.front_drivable_length = 0
        self.front_occupied = 0
        self.rear_drivable_length = 0
        self.driving_efficiency = 0
        self.closest_regular_object_id = 0
        self.closest_static_object_type = 0
        self.closest_moving_object_distance = 0
        self.closest_moving_object_type = 0
        self.closest_moving_object_id = 0

        self.projection_index = -1
        self.projection_x = -1
        self.projection_y = -1
        self.lateral_distance = -1
        self.dir_diff_signed = -1
        self.before_length = -1
        self.after_length = -1


class LanesOfInterest:
    def __init__(self):
        # self.points_x = []
        # self.points_y = []
        # self.points_num = 0
        # self.projection_x = 0
        # self.projection_y = 0
        self.lane_id = 0
        self.start_s = 0
        self.end_s = 0
        self.action_time = 0


class DecisionObstacle:
    def __init__(self, obstacle_msg, lane_list):
        self.id = obstacle_msg.id
        self.type = 0
        self.length = 0
        self.width = 0
        self.height = 0
        self.cur_bounding_points = []
        self.cur_velocity = 0
        self.cur_velocity_vec = None
        self.is_moving = 0

        # 时序问题，边界障碍物提取好后，去判断用什么车道作为当前车道，选好当前车道后，做规则障碍物向车道的投影
        self.if_tracked = 0
        self.detected_time = []
        self.history_center_points = []
        self.history_velocity = []
        self.history_velocity_vec = []
        self.history_heading = []

        self.cur_lane_id = 0

        # 存储每一个时刻运动信息到当前所在车道的投影信息
        self.s_record = []
        self.s_velocity = []
        self.l_velocity = []
        self.dir_diff = []
        self.lane_lateral_diff = []
        self.history_lane_ids = []

        self.target_lane_id = 0
        self.next_lane_id = 0
        self.intention = 0  # 0 for free move, 1 for lane keeping , 2 for lane change left, 3 for lane change right. 4 for passing zebra crossing.
        self.predicted_center_points = []
        self.predicted_headings = []

        self.sub_decision = AVOID_COLLISION
        self.safe_distance = MIN_SAFE_DISTANCE

        self.obstacle_update(obstacle_msg, lane_list)

        """
        # The type of the obstacle.
        # The value could be VEHICLE, BICYCLE, PEDESTRIAN, CONE, WATERHORSE, RAIL, OTHER, etc.
        string type
        
        # The size of the obstacle.
        # The length should be the distance in the direction of heading.
        # The width should be the distance perpendicular to the direction of heading.
        float32 length
        float32 width
        
        # The position of the boundary points on the obstacle.
        # In the relative coordinate system, "x" points to the front, "y" points to the left, "z" points upwards.
        # "z" may be not useful at this moment, and can be simply set as 0.
        # In the absolute coordinate system, it should be consistent with the map.
        geometry_msgs/Point32[] points
        
        # The estimated velocity of the obstacle.
        # Use the same coordinate system as for "points".
        geometry_msgs/Vector3 velocity
        
        # The predicted trajectory of the obstacle
        geometry_msgs/Point32[] predictedCenterPointsTrajectory
        
        # The predicted heading angle of the obstacle.
        float32[] predictedHeadings
        
        # Is the obstacle static or dynamic?
        # True for a dynamic obstacle.
        # False for a static obstacle.
        bool isMoving
        
        # The reaction towards the obstacle.
        int32 decision
        int32 NONE=0
        int32 OVERTAKE=1
        int32 GIVEWAY=2
        int32 BYPASS=3
        
        # A safe distance to be kept from the obstacle.
        float32 safeDistance
        """

    # update obstacles information, record the history movements of the obstacles.
    def obstacle_update(self, obstacle_msg, lane_list):
        # self.around_lanes = lane_list
        self.type = obstacle_msg.type
        self.if_tracked = 1
        self.cur_bounding_points = obstacle_msg.points

        self.cur_velocity_vec = obstacle_msg.velocity
        self.cur_velocity = math.sqrt(math.pow(obstacle_msg.velocity.x, 2) + math.pow(obstacle_msg.velocity.y, 2))
        if self.cur_velocity > 1:
            self.is_moving = True
        else:
            self.is_moving = False

        # record history trajectory for regular obstacles.
        if self.type == 'VEHICLE' or self.type == 'PEDESTRIAN' or self.type == 'BICYCLE':
            self.detected_time.append(obstacle_msg.detectedTime)
            # calculate center point
            center_point_x = np.mean([point_i.x for point_i in obstacle_msg.points])
            center_point_y = np.mean([point_i.y for point_i in obstacle_msg.points])
            center_point = Point32()
            center_point.x = center_point_x
            center_point.y = center_point_y
            center_point.z = 0
            self.history_center_points.append(center_point)
            self.history_velocity.append(self.cur_velocity)
            self.history_velocity_vec.append(
                [obstacle_msg.velocity.x, obstacle_msg.velocity.y, obstacle_msg.velocity.z])
            if self.is_moving:
                cur_heading = math.atan2(obstacle_msg.velocity.y, obstacle_msg.velocity.x)
                self.history_heading.append(cur_heading)

                project_heading = []
                project_lateral = []
                vec_heading = np.array([obstacle_msg.velocity.x, obstacle_msg.velocity.y])
                vec_lateral = np.array([-obstacle_msg.velocity.y, obstacle_msg.velocity.x])
                for point_i in obstacle_msg.points:
                    vec_point = np.array(
                        [point_i.x - center_point_x, point_i.y - center_point_y])
                    project_heading.append(np.dot(vec_heading, vec_point) / np.linalg.norm(vec_heading))
                    project_lateral.append(np.dot(vec_lateral, vec_point) / np.linalg.norm(vec_lateral))

                self.length = max(project_heading) - min(project_heading)
                self.width = max(project_lateral) - min(project_lateral)
            else:
                self.history_heading.append(0)
                self.length = 0
                self.width = 0

        # for other obstacles, update new information.
        else:
            self.detected_time.clear()
            self.history_center_points.clear()
            self.history_velocity.clear()
            self.history_velocity_vec.clear()
            self.history_heading.clear()
            # self.detected_time.append(obstacle_msg.detectedTime)

        if len(self.detected_time) > 5:
            self.detected_time = self.detected_time[-5:]
            self.history_heading = self.history_heading[-5:]
            self.history_velocity = self.history_velocity[-5:]
            self.history_velocity_vec = self.history_velocity_vec[-5:]
            self.history_center_points = self.history_center_points[-5:]

        # rospy.loginfo("start processing obstacle %d" % self.id)
        # project the obstacle to the lanes, and select current lane.
        if lane_list != None and (self.type == 'VEHICLE' or self.type == 'PEDESTRIAN' or self.type == 'BICYCLE'):
            # print(self.id, self.type, self.history_center_points)
            self.obstacle_projection(lane_list)
        else:
            self.lane_lateral_diff = []
            self.dir_diff = []
            self.s_record = []
            self.s_velocity = []
            self.l_velocity = []

        self.obstacle_behavior_initialization()

    # project the obstacle to the lanes, and select current lane.
    def obstacle_projection(self, lane_list):

        distance_threshold = 3
        direction_threshold = 30 / 180.0 * 3.14159265
        temp_distance = 0
        temp_direction_diff = 0
        last_lane_id = self.cur_lane_id
        last_target_lane_id = self.target_lane_id
        this_lane_id = 0
        lane_found_flag = False

        # for a moving object,

        # judge if there is a current lane. Yes, consider the current lane first, then the target lane. No, find the current lane.
        if last_lane_id != 0:
            if last_lane_id in lane_list.keys():
                cur_lane = lane_list[last_lane_id]
                points_x, points_y = [], []
                for j in cur_lane.points:
                    points_x.append(j.x)
                    points_y.append(j.y)
                points_num = len(points_x)
                points_x = np.array(points_x)
                points_y = np.array(points_y)
                result = lane_projection(points_x, points_y, points_num, self.history_center_points[-1].x,
                                         self.history_center_points[-1].y,
                                         self.history_heading[-1])
                temp_distance = result[3]
                temp_direction_diff = result[4]
                if abs(temp_distance) < distance_threshold:
                    self.cur_lane_id = last_lane_id
                    lane_found_flag = True
        # target lane
        if lane_found_flag == False and last_target_lane_id != 0:
            if last_target_lane_id in lane_list.keys():
                cur_lane = lane_list[last_target_lane_id]
                points_x, points_y = [], []
                for j in cur_lane.points:
                    points_x.append(j.x)
                    points_y.append(j.y)
                points_num = len(points_x)
                points_x = np.array(points_x)
                points_y = np.array(points_y)
                result = lane_projection(points_x, points_y, points_num, self.history_center_points[-1].x,
                                         self.history_center_points[-1].y,
                                         self.history_heading[-1])
                temp_distance = result[3]
                temp_direction_diff = result[4]
                if abs(temp_distance) < distance_threshold:
                    self.cur_lane_id = last_target_lane_id
                    lane_found_flag = True
        # if failed, find the current lane.
        if not lane_found_flag:
            min_distance = 1000
            for k in lane_list.keys():
                cur_lane = lane_list[k]
                points_x, points_y = [], []
                for j in cur_lane.points:
                    points_x.append(j.x)
                    points_y.append(j.y)
                points_num = len(points_x)
                points_x = np.array(points_x)
                points_y = np.array(points_y)
                result = lane_projection(points_x, points_y, points_num, self.history_center_points[-1].x,
                                         self.history_center_points[-1].y,
                                         self.history_heading[-1])
                temp_distance = result[3]
                temp_direction_diff = result[4]
                if abs(temp_distance) < distance_threshold and abs(
                        temp_distance) < min_distance:
                    min_distance = abs(temp_distance)
                    this_lane_id = k
            if this_lane_id != 0:
                self.cur_lane_id = this_lane_id
                lane_found_flag = True

        if not lane_found_flag:
            self.cur_lane_id = 0

        # if the current lane found, project history movement to the current lane.
        if lane_found_flag:
            if not self.history_lane_ids:
                self.history_lane_ids.append(self.cur_lane_id)
            else:
                if self.history_lane_ids[-1] != self.cur_lane_id:
                    self.history_lane_ids.append(self.cur_lane_id)
            # rospy.loginfo("obstacle id %d history lane ids %s" % (self.id, self.history_lane_ids))

            self.lane_lateral_diff = []
            self.dir_diff = []
            self.s_record = []
            self.s_velocity = []
            self.l_velocity = []

            cur_lane = lane_list[self.cur_lane_id]
            points_x, points_y = [], []
            for j in cur_lane.points:
                points_x.append(j.x)
                points_y.append(j.y)
            points_num = len(points_x)
            points_x = np.array(points_x)
            points_y = np.array(points_y)
            for i in range(len(self.detected_time)):
                result = lane_projection(points_x, points_y, points_num, self.history_center_points[i].x,
                                         self.history_center_points[i].y,
                                         self.history_heading[i])
                self.lane_lateral_diff.append(result[3])
                self.dir_diff.append(result[4])
                self.s_record.append(result[5])
                self.s_velocity.append(math.cos(result[4]) * self.history_velocity[i])
                self.l_velocity.append(math.sin(result[4]) * self.history_velocity[i])
            # rospy.loginfo("3333333333333333333333333333 %d %d %f %f"%(self.id, self.history_lane_ids[-1], self.s_velocity[-1], self.s_record[-1]))
        else:
            self.lane_lateral_diff = []
            self.dir_diff = []
            self.s_record = []
            self.s_velocity = []
            self.l_velocity = []


    def obstacle_behavior_initialization(self):
        if self.type == 'VEHICLE' or self.type == 'BICYCLE' or self.type == 'PEDESTRIAN':
            self.predicted_center_points = []
            self.predicted_headings = []
            self.predicted_headings.append(self.history_heading[-1])
            self.predicted_center_points.append(self.history_center_points[-1])
            self.sub_decision = AVOID_COLLISION
            self.safe_distance = MEDIUM_SAFE_DISTANCE
        else:
            pass

    """
    # predict the future intention of the obstacle
    def obstacle_intention_prediction(self, lane_list):

        if self.type == "VEHICLE" and "BICYCLE":
            if self.cur_lane_id != 0:
                if lane_list[self.cur_lane_id].turn == 0:
                    can_change_left_flag = lane_list[self.cur_lane_id].canChangeLeft
                    can_change_right_flag = lane_list[self.cur_lane_id].canChangeRight
                    lane_lateral_diff_cur = self.lane_lateral_diff[-1]
                    if len(self.lane_lateral_diff) > 10:
                        lane_lateral_diff_mean = sum(self.lane_lateral_diff[len(self.lane_lateral_diff) - 11:-1]) / 10
                    elif len(self.lane_lateral_diff) > 1:
                        lane_lateral_diff_mean = sum(self.lane_lateral_diff[0:-1]) / (len(self.lane_lateral_diff) - 1)
                    else:
                        lane_lateral_diff_mean = lane_lateral_diff_cur
                    if lane_lateral_diff_cur > lane_lateral_diff_mean and lane_lateral_diff_cur > 0 and self.l_velocity > 0 and can_change_left_flag == 1:
                        self.intention = 2
                        self.target_lane_id = lane_list[self.cur_lane_id].leftLaneId
                    elif lane_lateral_diff_cur < lane_lateral_diff_mean and lane_lateral_diff_cur < 0 and self.l_velocity < 0 and can_change_right_flag == 1:
                        self.intention = 3
                        self.target_lane_id = lane_list[self.cur_lane_id].rightLaneId
                    else:
                        self.intention = 1
                        self.target_lane_id = self.cur_lane_id
                else:
                    # intersection -- lane keeping
                    self.intention = 1
                    self.target_lane_id = self.cur_lane_id
                # select next lane id
                lead_to_ids = lane_list[self.target_lane_id].leadToIds
                vec_end_dir = np.array(
                    [lane_list[self.target_lane_id].points[-1].y - lane_list[self.target_lane_id].points[-3].y,
                     lane_list[self.target_lane_id].points[-1].x - lane_list[self.target_lane_id].points[-3].x])
                min_dir_diff = 100
                for i in range(len(lead_to_ids)):
                    if lead_to_ids[i] in lane_list.keys():
                        vec_start_dir = np.array(
                            [lane_list[lead_to_ids[i]].points[3].y - lane_list[lead_to_ids[i]].points[0].y,
                             lane_list[lead_to_ids[i]].points[3].x - lane_list[lead_to_ids[i]].points[0].x])
                        dir_diff = math.acos(np.dot(vec_end_dir, vec_start_dir) / (
                                    np.linalg.norm(vec_end_dir) * np.linalg.norm(vec_start_dir)))
                        if dir_diff < min_dir_diff:
                            min_dir_diff = dir_diff
                            self.next_lane_id = lead_to_ids[i]
            else:
                # free move predictor
                pass

        elif self.type == "PEDESTRIAN":
            pass

    # predict the future intention of the obstacle
    def obstacle_trajectory_prediction(self, lane_list):

        if self.target_lane_id in lane_list.keys():
            cur_lane = lane_list[self.target_lane_id]
            points_x, points_y = [], []
            for j in cur_lane.points:
                points_x.append(j.x)
                points_y.append(j.y)
            points_num = len(points_x)
            points_x = np.array(points_x)
            points_y = np.array(points_y)
            result = lane_projection(points_x, points_y, points_num, self.history_center_points[-1].x,
                                     self.history_center_points[-1].y,
                                     self.history_heading[-1])
            temp_distance = result[3]
        else:
            temp_distance = lane_list[self.cur_lane_id].width - abs(self.lane_lateral_diff[-1])

        approach_time = temp_distance / abs(self.l_velocity[-1])
        s_length = approach_time * abs(self.s_velocity[-1])

        predicted_points_list = []

        # polynomial curve-fitting
        predicted_x = []
        predicted_y = []
        for i in range(len(predicted_points_list)):
            predicted_x.append(predicted_points_list[i][0])
            predicted_y.append(predicted_points_list[i][1])
        points_num = len(predicted_points_list)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(x, y, color='m', linestyle='', marker='.')

        order = 5
        fitting_result = curve_fitting(predicted_x, predicted_y, points_num, order)

        xxa = np.arange(-1, 15, 0.01)
        yya = []
        for i in range(0, len(xxa)):
            yy = 0.0
            for j in range(0, order + 1):
                dy = 1.0
                for k in range(0, j):
                    dy *= xxa[i]
                dy *= fitting_result[j]
                yy += dy
            yya.append(yy)
    """

class TrafficLight:
    def __init__(self):
        self.light_type = 0
        self.color = 0
        self.blinking = False
        self.remain_time = 0
        self.position_x = 0
        self.position_y = 0

class ParkingSlot:
    def __init__(self):
        self.points = []
        self.origin = 0 # map_thing = 1, observe_thing = 2
        self.slot_type = 0 # parallel_slot = 1, vertical_slot = 2
        self.lane_id = 0
        self.lane_projection_point = []
        self.lane_projection_s = 0


# decision output message handler
decision_msg_pub = rospy.Publisher('decision_behavior', Decision, queue_size=1)
# re global planning service handler
rospy.wait_for_service('re_global_planning')
re_global_planning = rospy.ServiceProxy('re_global_planning', ReGlobalPlanning)
# mission done - service handler
rospy.wait_for_service('current_mission_finished')
current_mission_finished = rospy.ServiceProxy('current_mission_finished', MissionFinished)

@jit
def lane_projection(map_x, map_y, map_num, cur_x, cur_y, cur_yaw=0.0, type=0):
    """
    左负右正，cur_yaw 为弧度值
    project the point onto the current lane.
    :param map_x: a set of lane points (x coordination)
    :param map_y: a set of lane points (y coordination)
    :param map_num:
    :param cur_x: point x coordination
    :param cur_y: point y coordination
    :param cur_yaw: default value is 0, for mass points projection.
    :param type:
    :return:
        projection_x,
        projection_y,
        index, the projection point is in [index, index + 1]
        lateral_distance, lateral projection distance, positive on the left
        dir_diff_signed,
        before_length, longitudinal projection distance
        after_length
    """
    projection_x = 0
    projection_y = 0
    index = -1
    min_distance = 100000.0
    s_total = 0
    before_length = 0
    after_length = 0
    for i in range(map_num - 1):
        # a vector of one section in the way
        vec_section = np.array([map_x[i + 1] - map_x[i], map_y[i + 1] - map_y[i]])
        # a vector pointing from the start point of the section to the query point
        vec_point = np.array([cur_x - map_x[i], cur_y - map_y[i]])
        # calculate the projected point on the section as a 0~1 value with respect to the section length
        section_length = np.linalg.norm(vec_section)
        section_length_squared = section_length * section_length
        k = np.dot(vec_section, vec_point) / section_length_squared

        # if the projected point it outside the section, project it to the ends.
        if k >= 1.0:
            temp_projection_x = map_x[i + 1]
            temp_projection_y = map_y[i + 1]
        elif k <= 0.0:
            temp_projection_x = map_x[i]
            temp_projection_y = map_y[i]
        # else, project it perpendicularly.
        else:
            temp_projection_x = map_x[i] + k * vec_section[0]
            temp_projection_y = map_y[i] + k * vec_section[1]

        vec_offset = np.array([temp_projection_x - cur_x, temp_projection_y - cur_y])
        section_distance = np.linalg.norm(vec_offset)
        # record the minimum distance
        if section_distance < min_distance:
            min_distance = section_distance
            projection_x = temp_projection_x
            projection_y = temp_projection_y
            index = i

        #############################
        # 存在的问题，如果是环路，不知道是应该投影到这条路的头还是投影到这条路的尾

    # offset (lateral distance) with direction
    vec_section = np.array([map_x[index + 1] - map_x[index], map_y[index + 1] - map_y[index]])
    vec_point = np.array([cur_x - map_x[index], cur_y - map_y[index]])
    dir_temp = vec_section[1] * vec_point[0] - vec_section[0] * vec_point[1]
    if dir_temp < 0.0:
        dir_flag = -1.0
    elif dir_temp > 0.0:
        dir_flag = 1.0
    else:
        dir_flag = 0.0
    lateral_distance = dir_flag * min_distance

    # signed direction difference
    if cur_yaw == 0:
        dir_diff_signed = 0
    else:
        vec_map_dir = np.array([map_x[index + 1] - map_x[index], map_y[index + 1] - map_y[index]])
        vec_yaw_dir = np.array([math.cos(cur_yaw), math.sin(cur_yaw)])
        cos_value = np.dot(vec_map_dir, vec_yaw_dir) / (np.linalg.norm(vec_map_dir) * np.linalg.norm(vec_yaw_dir))
        if cos_value > 1:
            cos_value = 1
        if cos_value < -1:
            cos_value = -1
        dir_diff = math.acos(cos_value)
        dir_temp = vec_map_dir[1] * vec_yaw_dir[0] - vec_map_dir[0] * vec_yaw_dir[1]
        if dir_temp < 0.0:
            dir_flag = -1.0
        elif dir_temp > 0.0:
            dir_flag = 1.0
        else:
            dir_flag = 0.0
        dir_diff_signed = dir_flag * dir_diff

    for j in range(0, index):
        before_length += math.sqrt(math.pow(map_x[j + 1] - map_x[j], 2) + math.pow(map_y[j + 1] - map_y[j], 2))
    for j in range(index + 1, map_num - 1):
        after_length += math.sqrt(math.pow(map_x[j + 1] - map_x[j], 2) + math.pow(map_y[j + 1] - map_y[j], 2))
    before_length += math.sqrt(math.pow(projection_x - map_x[index], 2) + math.pow(projection_y - map_y[index], 2))
    after_length += math.sqrt(
        math.pow(projection_x - map_x[index + 1], 2) + math.pow(projection_y - map_y[index + 1], 2))

    if abs(projection_x - map_x[-1]) < EPS:
        index = map_num - 1

    return projection_x, projection_y, index, lateral_distance, dir_diff_signed, before_length, after_length


def road_callback(road_msg):
    global road_updated_flag
    if road_updated_flag and user_data_copied_flag:
        road_updated_flag = 0
        global lane_list
        road_data = road_msg
        lane_list = {}  # {'id':'lane'}
        for k in range(len(road_data.lanes)):
            lane_list[road_data.lanes[k].id] = road_data.lanes[k]
        road_updated_flag = 1
            # rospy.loginfo('map_data_updated')


def global_pose_callback(global_pose_msg):
    global pose_updated_flag
    if pose_updated_flag and user_data_copied_flag:
        pose_updated_flag = 0
        global global_pose_data
        global_pose_data = global_pose_msg
        global_pose_data.mVf = abs(global_pose_data.mVf)
        pose_updated_flag = 1
        # rospy.loginfo('pose_data_updated')


def lights_callback(lights_msg):
    global lights_updated_flag
    if lights_updated_flag and user_data_copied_flag:
        lights_updated_flag = 0
        global lights_list
        lights_list = {}
        direction_set = [STRAIGHT, LEFT, RIGHT, U_TURN]
        # observed_direction_set = []
        other_light = None
        # the definition align with road turn definition
        for light in lights_msg.lights:
            # observed_direction_set.append(light.lightType)
            if light.lightType == ARROW_LEFT:
                lights_list[LEFT] = light
                lights_list[U_TURN] = light
            if light.lightType == ARROW_RIGHT:
                lights_list[RIGHT] = light
            if light.lightType == ARROW_DOWN or light.lightType == ARROW_UP:
                lights_list[STRAIGHT] = light
            if light.lightType == CIRCLE:
                other_light = light

        if lights_list.keys():
            for i in direction_set:
                if i not in lights_list.keys():
                    lights_list[i] = other_light

        lights_updated_flag = 1
        # rospy.loginfo('lights_data_updated')


def signs_callback(signs_msg):
    global signs_updated_flag
    if signs_updated_flag and user_data_copied_flag:
        signs_updated_flag = 0
        global signs_data
        signs_data = signs_msg
        rospy.loginfo('signs_data_updated')
        signs_updated_flag = 1


def obstacles_callback(obstacles_msg):

    global obstacle_updated_flag
    if obstacle_updated_flag and user_data_copied_flag:
        obstacle_updated_flag = 0
        a = rospy.get_time()
        # rospy.loginfo('start receiving obstacle data ------ at time %f' % rospy.get_time())
        global obstacles_list
        # record road data of the current moment.
        temp_lane_info = lane_list
        # reset the if_tracked tag
        for k in obstacles_list.keys():
            obstacles_list[k].if_tracked = 0

        for i in range(len(obstacles_msg.obstacles)):
            # update old obstacles
            if obstacles_msg.obstacles[i].id in obstacles_list.keys():
                obstacles_list[obstacles_msg.obstacles[i].id].obstacle_update(obstacles_msg.obstacles[i], temp_lane_info)
            # generate new obstacle
            else:
                temp_obstacle = DecisionObstacle(obstacles_msg.obstacles[i], temp_lane_info)
                obstacles_list[obstacles_msg.obstacles[i].id] = temp_obstacle

        for k in list(obstacles_list.keys()):
            if obstacles_list[k].if_tracked == 0:
                del obstacles_list[k]

        # rospy.loginfo('obstacle process done ------ at time %f' % rospy.get_time())
        b = rospy.get_time()
        process_time = b - a
        if process_time > 0.2:
            rospy.loginfo('obstacle process duration %f' % (b - a))
        obstacle_updated_flag = 1

    # for i in obstacles_list.keys():
    #     if obstacles_list[i].cur_lane_id > 0:
    #         print(obstacles_list[i].cur_lane_id, obstacles_list[i].s_record[-1], obstacles_list[i].s_velocity[-1], len(obstacles_list[i].detected_time))
    # rospy.loginfo('obstacles_data_updated')


def parking_slot_callback(things_msg):
    global parking_slot_updated_flag
    if parking_slot_updated_flag and user_data_copied_flag:
        parking_slot_updated_flag = 0
        global parking_slots_list
        things_data = things_msg
        parking_slots_list = {}
        # type is parkingArea or parkingSlot
        for k in things_data.things:
            if k.type == "parkingSlot":
                temp_parking_slot = ParkingSlot()
                temp_parking_slot.id = k.id
                temp_parking_slot.points = k.points
                temp_parking_slot.lane_id = k.laneId
                temp_parking_slot.lane_projection_point = k.laneProjectedPoint
                first_edge_distance = math.sqrt(math.pow(k.points[0].x - k.points[1].x, 2) + math.pow(k.points[0].y - k.points[1].y, 2))
                second_edge_distance = math.sqrt(math.pow(k.points[2].x - k.points[1].x, 2) + math.pow(k.points[2].y - k.points[1].y, 2))
                if first_edge_distance < second_edge_distance:
                    temp_parking_slot.slot_type = PARALLEL_SLOT
                else:
                    temp_parking_slot.slot_type = VERTICAL_SLOT
                parking_slots_list[k.id] = temp_parking_slot
            # if k.type == "parkingArea":
            #     parking_area_list[k.id] = k.points
#        print(parking_slots_list.keys())
        parking_slot_updated_flag = 1


def map_things_callback(things_msg):
    global map_things_updated_flag
    if map_things_updated_flag and user_data_copied_flag:
        map_things_updated_flag = 0
        global parking_area_list
        things_data = things_msg
        for k in things_data.things:
            if k.type == "parkingArea":
                parking_area_list[k.id] = k.points

        global parking_slots_list
        parking_slots_list = {}
        # type is parkingArea or parkingSlot
        for k in things_data.things:
            if k.type == "parkingSlot":
                temp_parking_slot = ParkingSlot()
                temp_parking_slot.id = k.id
                temp_parking_slot.points = k.points
                temp_parking_slot.lane_id = k.laneId
                temp_parking_slot.lane_projection_point = k.laneProjectedPoint
                first_edge_distance = math.sqrt(
                    math.pow(k.points[0].x - k.points[1].x, 2) + math.pow(k.points[0].y - k.points[1].y, 2))
                second_edge_distance = math.sqrt(
                    math.pow(k.points[2].x - k.points[1].x, 2) + math.pow(k.points[2].y - k.points[1].y, 2))
                if first_edge_distance < second_edge_distance:
                    temp_parking_slot.slot_type = PARALLEL_SLOT
                else:
                    temp_parking_slot.slot_type = VERTICAL_SLOT
                parking_slots_list[k.id] = temp_parking_slot
            # if k.type == "parkingArea":
            #     parking_area_list[k.id] = k.points

        map_things_updated_flag = 1


# mission的type有值，找到mission所在的lane id，当车开到目标车道时，开始考虑任务
# 如果是park任务，找到mission的thingid在thing中对应的type
# 如果是parkingArea，则从mission的位置开始从things中选择目标车位，
# 如果是parkingslot，则停到对应的id的车位

def mission_callback(mission_msg):
    global mission_ahead
    mission_ahead = MissionAhead()
    # road, park, point
    mission_ahead.missionType = mission_msg.destinationTypes[0]
    mission_ahead.missionLaneIds = mission_msg.firstMissionDestinationWayIds
    mission_ahead.missionLocationX = mission_msg.firstDestinationMapX
    mission_ahead.missionLocationY = mission_msg.firstDestinationMapY
    mission_ahead.missionIndex = mission_msg.currentMissionIndex
    mission_ahead.missionThingId = mission_msg.firstDestinationThingId


def listener():
    # 注意node的名字得独一无二，但是topic的名字得和你想接收的信息的topic一样！
    # rospy.init_node('listener', anonymous = True)

    # Subscriber函数第一个参数是topic的名称，第二个参数是接受的数据类型，第三个参数是回调函数的名称
    rospy.Subscriber("global_pose", GlobalPose, global_pose_callback, queue_size=1, buff_size=5000000)
    rospy.Subscriber("map_road", Road, road_callback, queue_size=1, buff_size=5000000)
    rospy.Subscriber("obstacles", Obstacles, obstacles_callback, queue_size=1, buff_size=5000000)
    rospy.Subscriber("traffic_lights", Lights, lights_callback, queue_size=1, buff_size=5000000)
    rospy.Subscriber("traffic_signs", Signs, signs_callback, queue_size=1, buff_size=5000000)
    rospy.Subscriber("map_missions", Missions, mission_callback, queue_size=1, buff_size=5000000)
#    rospy.Subscriber("parking_slot", Things, parking_slot_callback, queue_size=1, buff_size=5000000)
    rospy.Subscriber("map_thing", Things, map_things_callback, queue_size=1, buff_size=5000000)


    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()
    # 只 spin 有 callback 的语句


def planning_feedback_callback(plan_request):
    rospy.loginfo("5555555555555555555555555555555555")
    rospy.loginfo("planning feedback: %d" % plan_request.planningFeedback)
    global planning_feedback
    planning_feedback = plan_request.planningFeedback
    return PlanningFeedbackResponse(1)


def server():
    rospy.Service('planning_feedback', PlanningFeedback, planning_feedback_callback)


def compute_mean(nums, start, end):
    sum = 0
    count = 0
    stop = min(end + 1, len(nums))
    # print(stop)
    for i in range(start, stop, 1):
        sum += nums[i]
        count += 1
    # print(count)
    if count == 0:
        mean = 0
    else:
        mean = sum / count
    return mean


def curve_fitting(points_x, points_y, points_num, order):
    list_x = []
    for xx in points_x:
        list_line = []
        for i in range(0, order + 1):
            list_line.append(xx ** i)
        list_x.append(list_line)
    mat_x = np.reshape(list_x, (points_num, order + 1))

    mat_y = np.reshape(points_y, (points_num, 1))

    mat_w = np.zeros((points_num, points_num))
    for i in range(points_num):
        # mat_w[i, i] = 10000 - i * 9900 / order
        # mat_w[i, i] = 1/mat_w[i,i]
        # mat_w[i, i] = i + 1
        mat_w[i, i] = 1

    mat_m = np.dot(np.transpose(mat_x), mat_w)
    fitting_result = np.dot(np.dot(np.linalg.inv(np.dot(mat_m, mat_x)), mat_m), mat_y)
    # temp_result = np.linalg.inv(np.dot(np.transpose(mat_x), mat_x))
    # fitting_result = np.dot(np.dot(temp_result, np.transpose(mat_x)), mat_y)
    return fitting_result


def user_data_updater(user_data):
    global user_data_copied_flag
    if obstacle_updated_flag and road_updated_flag and lights_updated_flag and signs_updated_flag and pose_updated_flag and map_things_updated_flag and parking_slot_updated_flag:
        user_data_copied_flag = 0
        rospy.loginfo('updating data ------ at time %f' % rospy.get_time())
        user_data.lane_list = copy.deepcopy(lane_list)
        user_data.pose_data = copy.deepcopy(global_pose_data)
        user_data.obstacles_list = copy.deepcopy(obstacles_list)
        user_data.signs_data = copy.deepcopy(signs_data)
        user_data.lights_list = copy.deepcopy(lights_list)
        user_data.parking_slots_list = copy.deepcopy(parking_slots_list)
        user_data_copied_flag = 1
    # print('data updated.')


def parking_slot_choose_decider():
    pass


def current_lane_selector(lane_list, pose_data):
    # 投影信息用字典存储
    available_lanes = {}
    id_list, offset, dir_diff, before_length, after_length = [], [], [], [], []
    project_x, project_y, project_index = [], [], []
    points_num_set = []
    # rospy.loginfo(lane_list.keys())
    for lane_index in lane_list.keys():
        temp_lane = lane_list[lane_index]
        id_list.append(lane_index)
        points_x, points_y = [], []
        for j in range(len(temp_lane.points)):
            points_x.append(temp_lane.points[j].x)
            points_y.append(temp_lane.points[j].y)
        points_num = len(points_x)
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        result = lane_projection(points_x, points_y, points_num, pose_data.mapX, pose_data.mapY, pose_data.mapHeading)
        offset.append(result[3])
        dir_diff.append(result[4])
        before_length.append(result[5])
        after_length.append(result[6])
        project_x.append(result[0])
        project_y.append(result[1])
        project_index.append(result[2])
        points_num_set.append(points_num)

        temp_drivable_lane = DrivableLanes()
        temp_drivable_lane.projection_x = result[0]
        temp_drivable_lane.projection_y = result[1]
        temp_drivable_lane.projection_index = result[2]
        temp_drivable_lane.lateral_distance = result[3]
        temp_drivable_lane.dir_diff_signed = result[4]
        temp_drivable_lane.before_length = result[5]
        temp_drivable_lane.after_length = result[6]
        available_lanes[lane_index] = temp_drivable_lane

    # choose current lane id and index
    DIR_THRESHOLD = 90.0 / 180.0 * 3.14159265
    OFFSET_THRESHOLD = 2.0
    min_offset = 2.0
    cur_lane_index = -1
    cur_lane_id = -1
    count = 0

    priority_id_index_set = []

    # 选择当前车道(找全局规划)
    # 先找距离处于较小范围内，优先级较高的第一条车道
    # 如果找不到，找距离最小的一条车道
    for i in range(len(id_list)):
        abs_offset = abs(offset[i])
        # 附近车道，距离最小，方向偏差小，优先级高（2），不是车道末段
        if abs_offset < OFFSET_THRESHOLD \
                and lane_list[id_list[i]].priority == 2 \
                and project_index[i] < points_num_set[i] - 1:
            count += 1
            priority_id_index_set.append([id_list[i], i])

    if count == 0:
        for i in range(len(id_list)):
            abs_offset = abs(offset[i])
            # 附近车道，距离最小，方向偏差小，优先级为 1，不是车道末段
            if abs_offset < OFFSET_THRESHOLD \
                    and lane_list[id_list[i]].priority == 1 \
                    and project_index[i] < points_num_set[i] - 1:
                count += 1
                priority_id_index_set.append([id_list[i], i])

    if count == 0:
        min_offset_id = -1
        min_offset_index = -1
        for i in range(len(id_list)):
            abs_offset = abs(offset[i])
            if abs_offset < min_offset and project_index[i] < points_num_set[i] - 1:
                count = 1
                min_offset = abs_offset
                min_offset_id = id_list[i]
                min_offset_index = i
        if min_offset_id != -1:
            priority_id_index_set.append([min_offset_id, min_offset_index])

    if count != 0:
        # min_id = 10000
        # min_id_index = -1
        priority_id_index_set.sort()

        if history_lane_ids != []:
            for i in range(count):
                if history_lane_ids[-1] == priority_id_index_set[i][0] and abs(dir_diff[priority_id_index_set[i][1]]) < DIR_THRESHOLD:
                    cur_lane_id = priority_id_index_set[i][0]
                    cur_lane_index = priority_id_index_set[i][1]

        if cur_lane_id == -1:
            for i in range(count):
                if abs(dir_diff[priority_id_index_set[i][1]]) < DIR_THRESHOLD:
                    cur_lane_id = priority_id_index_set[i][0]
                    cur_lane_index = priority_id_index_set[i][1]
                    break

        if cur_lane_id == -1:
            cur_lane_id = priority_id_index_set[0][0]
            cur_lane_index = priority_id_index_set[0][1]


    cur_lane_info = CurrentLaneInfo()
    if cur_lane_id != -1:
        cur_lane_info.cur_lane_id = cur_lane_id
        try:
            cur_lane_info.left_lane_id = lane_list[cur_lane_id].leftLaneId
        except:
            pass
        try:
            cur_lane_info.right_lane_id = lane_list[cur_lane_id].rightLaneId
        except:
            pass
        cur_lane_info.projection_x = project_x[cur_lane_index]
        cur_lane_info.projection_y = project_y[cur_lane_index]
        cur_lane_info.projection_index = project_index[cur_lane_index]
        cur_lane_info.lateral_distance = offset[cur_lane_index]
        cur_lane_info.dir_diff_signed = dir_diff[cur_lane_index]
        cur_lane_info.before_length = before_length[cur_lane_index]
        cur_lane_info.after_length = after_length[cur_lane_index]

        cur_lane_info.next_stop_type = lane_list[cur_lane_id].stopType
        if lane_list[cur_lane_id].stopType != 0:
            stop_line_x = lane_list[cur_lane_id].nextStop.x
            stop_line_y = lane_list[cur_lane_id].nextStop.y
            cur_lane_info.next_stop_x = stop_line_x
            cur_lane_info.dist_to_next_stop = math.sqrt(
                math.pow(stop_line_x - pose_data.mapX, 2) + pow(stop_line_y - pose_data.mapY, 2))

        cur_lane_info.dist_to_next_road = after_length[cur_lane_index]
        cur_lane_info.cur_turn_type = lane_list[cur_lane_id].turn

        # print(cur_lane_info.next_stop_type, cur_lane_info.dist_to_next_stop, cur_lane_info.dist_to_next_road, cur_lane_info.cur_turn_type)

        cur_lane_info.can_change_left = lane_list[cur_lane_id].canChangeLeft
        cur_lane_info.can_change_right = lane_list[cur_lane_id].canChangeRight
        cur_lane_info.speed_lower_limit = lane_list[cur_lane_id].speedLowerLimit / 3.6
        cur_lane_info.speed_upper_limit = lane_list[cur_lane_id].speedUpperLimit / 3.6

        cur_lane_info.cur_priority = lane_list[cur_lane_id].priority

        temp_id = cur_lane_id
        for i in range(10):
            temp_left_id = lane_list[temp_id].leftLaneId
            try:
                cur_lane_info.left_priority.append(lane_list[temp_left_id].priority)
                temp_id = temp_left_id
            except:
                break

        temp_id = cur_lane_id
        for i in range(10):
            temp_right_id = lane_list[temp_id].rightLaneId
            try:
                cur_lane_info.right_priority.append(lane_list[temp_right_id].priority)
                temp_id = temp_right_id
            except:
                break
    rospy.loginfo('current lane id %d' % cur_lane_info.cur_lane_id)
    rospy.loginfo('current lane priority %d' % cur_lane_info.cur_priority)
    return cur_lane_info, available_lanes


def available_lanes_selector(lane_list, pose_data, obstacles_list, cur_lane_info, available_lanes):
    """
    返回车道的可行驶距离。可行驶距离由距离最近的挡住车道正常行驶宽度的静态障碍物决定。
    对于对向车道,正向可行距离和反向可行距离都是需要的。
    或许可以同时用当前车道的静态障碍物做能否左右变道的判断
    这里粗略的选择了可行驶车道
    考虑进动态障碍物和静态障碍物
    :return: a set of available lanes
    """
    rospy.loginfo("available lane list %s " % list(available_lanes.keys()))
    temp_can_change_left = 1
    temp_can_change_right = 1
    # 以后可以加上感兴趣车道的选取，只把障碍物投影到感兴趣车道上
    for lane_index in available_lanes.keys():
        temp_lane = lane_list[lane_index]
        # if temp_lane.priority <= 0:
        #     continue
        points_x, points_y = [], []
        for j in range(len(temp_lane.points)):
            points_x.append(temp_lane.points[j].x)
            points_y.append(temp_lane.points[j].y)
        points_num = len(points_x)
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        vehicle_s = available_lanes[lane_index].before_length
        vehicle_l = available_lanes[lane_index].lateral_distance
        # lon_distance_interest = vehicle_s - MIN_TURNING_RADIUS
        # left_margin = -1 / 2 * temp_lane.width + (temp_lane.width - 1.2 * VEHICLE_WIDTH)
        # right_margin = 1 / 2 * temp_lane.width - (temp_lane.width - 1.2 * VEHICLE_WIDTH)
        # left_margin = 0.5 * LANE_WIDTH_BASE - 1.2 * VEHICLE_WIDTH
        # right_margin = -0.5 * LANE_WIDTH_BASE + 1.2 * VEHICLE_WIDTH
        left_margin = -0.6 * VEHICLE_WIDTH
        right_margin = 0.6 * VEHICLE_WIDTH

        # front_drivable_s = 100000
        # rear_drivable_s = -100000
        front_drivable_s = available_lanes[lane_index].after_length + available_lanes[lane_index].before_length
        rear_drivable_s = 0
        # 自车处于当前车道上, 计算车前和车后可行驶长度

        lane_efficiency = temp_lane.speedUpperLimit / 3.6

        temp_efficiency = 120
        moving_object_id = 0
        moving_object_type = 0
        moving_object_s = front_drivable_s
        static_object_type = 0
        regular_obstacle_id = 0
        # define efficiency by the closest dynamic obstacle to the vehicle

        for obstacle_index in obstacles_list.keys():
            temp_obstacle = obstacles_list[obstacle_index]

            lateral_range = []
            longitudinal_range = []
            for point in temp_obstacle.cur_bounding_points:
                result = lane_projection(points_x, points_y, points_num,
                                         point.x,
                                         point.y)
                lateral_range.append(result[3])
                longitudinal_range.append(result[5])
            lateral_min = min(lateral_range)
            lateral_max = max(lateral_range)
            longitudinal_min = min(longitudinal_range)
            longitudinal_max = max(longitudinal_range)
            # longitudinal condition
            if longitudinal_max > vehicle_s:
                if not (lateral_max < left_margin or lateral_min > right_margin):
                    if longitudinal_min < front_drivable_s:
                        front_drivable_s = longitudinal_min
                        if temp_obstacle.type == 'VEHICLE' or temp_obstacle.type == 'BICYCLE' or temp_obstacle.type == 'PEDESTRIAN':
                            # moving_object_s = temp_obstacle.s_record[-1]
                            # moving_object_type = temp_obstacle.type
                            regular_obstacle_id = temp_obstacle.id
                            if temp_obstacle.cur_lane_id > 0:
                                temp_efficiency = temp_obstacle.s_velocity[-1]
                            else:
                                result = lane_projection(points_x, points_y, points_num,
                                                         temp_obstacle.history_center_points[-1].x,
                                                         temp_obstacle.history_center_points[-1].y,
                                                         temp_obstacle.history_heading[-1])
                                temp_efficiency = math.cos(result[4]) * temp_obstacle.history_velocity[-1]
                        else:
                            # static_object_type = temp_obstacle.type
                            temp_efficiency = 0

            elif longitudinal_min < vehicle_s:
                if not (lateral_max < left_margin or lateral_min > right_margin):
                    if longitudinal_max > rear_drivable_s:
                        rear_drivable_s = longitudinal_max

            if lane_index == cur_lane_info.cur_lane_id:
                if not (longitudinal_max < vehicle_s or longitudinal_min > (vehicle_s + 0.5 * LANE_CHANGE_BASE_LENGTH)):
                    if not (lateral_max < (-0.5 * LANE_WIDTH_BASE - VEHICLE_WIDTH) or lateral_min > -0.5 * LANE_WIDTH_BASE):
                        temp_can_change_left = 0
                    if not (lateral_max < 0.5 * LANE_WIDTH_BASE or lateral_min > (0.5 * LANE_WIDTH_BASE + VEHICLE_WIDTH)):
                        temp_can_change_right = 0

            # if temp_obstacle.is_moving:
            #     # 找车前最近的在车道上的动态障碍物的速度
            #     if temp_obstacle.cur_lane_id == lane_index:
            #         if vehicle_s < temp_obstacle.s_record[-1] < moving_object_s:
            #             moving_object_s = temp_obstacle.s_record[-1]
            #             temp_efficiency = temp_obstacle.s_velocity[-1]
            #             moving_object_type = temp_obstacle.type
            #             moving_object_id = temp_obstacle.id

        if front_drivable_s < vehicle_s:
            front_drivable_length = 0
        else:
            front_drivable_length = front_drivable_s - vehicle_s

        if front_drivable_length > OBSERVE_RANGE:
            front_drivable_length = OBSERVE_RANGE

        if rear_drivable_s > vehicle_s:
            rear_drivable_length = 0
        else:
            rear_drivable_length = vehicle_s - rear_drivable_s

        if rear_drivable_length > OBSERVE_RANGE:
            rear_drivable_length = OBSERVE_RANGE

        # moving_obstacle_distance = moving_object_s - vehicle_s
        # moving_obstacle_distance = min(front_drivable_length , moving_obstacle_distance)

        # if min(available_lanes[lane_index].after_length, OBSERVE_RANGE) - front_drivable_length > EPS:

        # if front_drivable_length <= moving_obstacle_distance:
        #     temp_efficiency = 0
        # else:
        #     temp_efficiency = temp_efficiency
        # if moving_obstacle_distance < front_drivable_length:
        #     lane_efficiency = temp_efficiency

        # if min(available_lanes[lane_index].after_length, OBSERVE_RANGE) - front_drivable_length > EPS :
        #     lane_efficiency = 0
            # 如果有静态障碍物限制可行驶距离，车道效率置0
        lane_efficiency = min(temp_efficiency, lane_efficiency)

        temp_drivable_lane = available_lanes[lane_index]
        temp_drivable_lane.front_drivable_length = front_drivable_length
        temp_drivable_lane.rear_drivable_length = rear_drivable_length
        temp_drivable_lane.driving_efficiency = lane_efficiency
        temp_drivable_lane.closest_regular_object_id = regular_obstacle_id
        # temp_drivable_lane.closest_static_object_type = static_object_type
        # temp_drivable_lane.closest_moving_object_distance = moving_obstacle_distance
        # temp_drivable_lane.closest_moving_object_type = moving_object_type
        # temp_drivable_lane.closest_moving_object_id = moving_object_id

        # print(lane_index, temp_drivable_lane.closest_moving_object_id, temp_drivable_lane.closest_moving_object_type,
        #       temp_drivable_lane.closest_moving_object_distance, temp_drivable_lane.closest_static_object_type,
        #       temp_drivable_lane.driving_efficiency, temp_drivable_lane.front_drivable_length,
        #       temp_drivable_lane.after_length,temp_drivable_lane.front_drivable_length)

    cur_lane_info.can_change_left = temp_can_change_left and cur_lane_info.can_change_left
    cur_lane_info.can_change_right = temp_can_change_right and cur_lane_info.can_change_right

    return available_lanes, cur_lane_info


def target_lane_selector(lane_list, pose_data, scenario, cur_lane_info, available_lanes):
    """
    根据场景选择当前目标车道，
    如果小于期望预瞄距离，则进行下一条车道的拼接
    """
    target_lane_id = -1
    # 当前没有目标车道或者当前车道优先级不为正,为了切入车道，只需在场景开始时选择一次
    if scenario == "merge":
        # 暂时先考虑最近的车道,需满足有足够的行驶距离，并且优先级为正
        lane_offset_id_list = []
        for lane_index in available_lanes.keys():
            lane_offset_id_list.append([abs(available_lanes[lane_index].lateral_distance), lane_index])
        lane_offset_id_list.sort()
        for i in range(len(lane_offset_id_list)):
            if lane_offset_id_list[i][0] > 20:
                break
            if available_lanes[lane_offset_id_list[i][1]].front_drivable_length > EPS \
                    and lane_list[lane_offset_id_list[i][1]].priority > 0:
                target_lane_id = lane_offset_id_list[i][1]
                break

    # 当前处于正常行驶状态，为了提升行驶效率，而选择目标车道
    elif scenario == "lane_follow":
        selectable_lanes = [cur_lane_info.cur_lane_id, cur_lane_info.left_lane_id, cur_lane_info.right_lane_id]
        if cur_lane_info.left_lane_id not in available_lanes.keys():
            selectable_lanes[1] = 0
        if cur_lane_info.right_lane_id not in available_lanes.keys():
            selectable_lanes[2] = 0
        lane_reward = []
        lane_priority = []
        can_change_constraint = [1, cur_lane_info.can_change_left, cur_lane_info.can_change_right]

        try:
            left_priority = cur_lane_info.left_priority[0]
        except:
            left_priority = 0
        try:
            right_priority = cur_lane_info.right_priority[0]
        except:
            right_priority = 0
        left_found_2, right_found_2 = 0, 0
        left_index, right_index = 11, 11
        for i in range(10):
            if left_found_2 == 0 and i < len(cur_lane_info.left_priority):
                if cur_lane_info.left_priority[i] == 2:
                    left_found_2 = 1
                    left_index = i
            if right_found_2 == 0 and i < len(cur_lane_info.right_priority):
                if cur_lane_info.right_priority[i] == 2:
                    right_found_2 = 1
                    right_index = i
        if left_found_2 and right_found_2:
            if left_index <= right_index:
                left_priority = 2
            else:
                right_priority = 2
        elif left_found_2:
            left_priority = 2
        elif right_found_2:
            right_priority = 2

        temp_priority_list = []
        # 与剩余路径长短成反比的优先级系数
        if cur_lane_info.dist_to_next_road <= 100:
            lane_priority += [cur_lane_info.cur_priority, left_priority, right_priority]
        else:
            if cur_lane_info.cur_priority == 2:
                temp_priority_list.append(1 + 100 / cur_lane_info.dist_to_next_road)
            else:
                temp_priority_list.append(cur_lane_info.cur_priority)
            if left_priority == 2:
                temp_priority_list.append(1 + 100 / cur_lane_info.dist_to_next_road)
            else:
                temp_priority_list.append(left_priority)
            if right_priority == 2:
                temp_priority_list.append(1 + 100 / cur_lane_info.dist_to_next_road)
            else:
                temp_priority_list.append(left_priority)
            lane_priority += temp_priority_list


        lane_drivable_length = []
        lane_drivable_length_ratio = []
        lane_efficiency_ratio = []
        lane_efficiency = []
        for i in range(len(selectable_lanes)):
            if selectable_lanes[i] > 0:
                efficiency = available_lanes[selectable_lanes[i]].driving_efficiency / (
                            lane_list[selectable_lanes[i]].speedUpperLimit / 3.6)
                lane_efficiency.append(available_lanes[selectable_lanes[i]].driving_efficiency)
                lane_efficiency_ratio.append(efficiency)
                if available_lanes[selectable_lanes[i]].after_length < EPS:
                    free_space = 0
                else:
                    free_space = available_lanes[selectable_lanes[i]].front_drivable_length / min(available_lanes[
                    selectable_lanes[i]].after_length, OBSERVE_RANGE)
                lane_drivable_length.append(available_lanes[selectable_lanes[i]].front_drivable_length)
                lane_drivable_length_ratio.append(free_space)

                lane_reward.append((0.5 * efficiency + 0.5 * free_space) * lane_priority[i] * can_change_constraint[i])
            else:
                lane_reward.append(0)
                lane_efficiency_ratio.append(0)
                lane_drivable_length_ratio.append(0)
                lane_efficiency.append(0)
                lane_drivable_length.append(0)
        rospy.loginfo("cur left right lane ids %s" % list(selectable_lanes))
        rospy.loginfo("lane_priority %s" % list(lane_priority))
        rospy.loginfo("can_change_constraint %s" % list(can_change_constraint))
        rospy.loginfo("lane_reward %s" % list(lane_reward))
        rospy.loginfo("lane_efficiency %s" % list(lane_efficiency))
        rospy.loginfo("lane_efficiency_ratio %s" % list(lane_efficiency_ratio))
        rospy.loginfo("lane_drivable_length %s" % list(lane_drivable_length))
        rospy.loginfo("lane_drivable_length_ratio %s" % list(lane_drivable_length_ratio))

        rospy.loginfo("dis to next road %f" % cur_lane_info.dist_to_next_road)

        # 先判断是否需要强制性变道到优先级高的车道
        if 20 < cur_lane_info.dist_to_next_road < 100:
            # 条件有待确定
            if lane_priority[0] == 2:
                if available_lanes[cur_lane_info.cur_lane_id].driving_efficiency > 0.2 * (
                        lane_list[cur_lane_info.cur_lane_id].speedUpperLimit / 3.6) or \
                        available_lanes[selectable_lanes[0]].front_drivable_length > OBSERVE_RANGE - LANE_CHANGE_BASE_LENGTH:
                    target_lane_id = cur_lane_info.cur_lane_id
                else:
                    if lane_reward[1] > lane_reward[0] * 2 + EPS and lane_reward[2] > lane_reward[0] * 2 + EPS:
                        if lane_reward[2] > lane_reward[1] * 1.2 + EPS and lane_drivable_length[2] > EPS + LANE_CHANGE_BASE_LENGTH:
                            target_lane_id = cur_lane_info.right_lane_id
                        elif lane_reward[2] <= lane_reward[1] * 1.2+ EPS and lane_drivable_length[1] > EPS + LANE_CHANGE_BASE_LENGTH:
                            target_lane_id = cur_lane_info.left_lane_id
                    elif lane_reward[1] > lane_reward[0] * 2 + EPS and lane_drivable_length[1] > EPS + LANE_CHANGE_BASE_LENGTH:
                        target_lane_id = cur_lane_info.left_lane_id
                    elif lane_reward[2] > lane_reward[0] * 2 + EPS and lane_drivable_length[2] > EPS + LANE_CHANGE_BASE_LENGTH:
                        target_lane_id = cur_lane_info.right_lane_id
                    else:
                        target_lane_id = cur_lane_info.cur_lane_id
            else:
                max_lane_reward = max(lane_reward)
                if max_lane_reward - 0 > EPS:
                    if lane_reward[1] == max(lane_reward):
                        target_lane_id = cur_lane_info.left_lane_id
                    elif lane_reward[2] == max(lane_reward):
                        target_lane_id = cur_lane_info.right_lane_id
                else:
                    target_lane_id = cur_lane_info.cur_lane_id
        elif cur_lane_info.dist_to_next_road <= 20:
            target_lane_id = cur_lane_info.cur_lane_id

        elif cur_lane_info.dist_to_next_road >= 100:
            # 先判断当前车道的行驶效率
            if available_lanes[cur_lane_info.cur_lane_id].driving_efficiency > 0.6 * (
                    lane_list[cur_lane_info.cur_lane_id].speedUpperLimit / 3.6) or \
                    available_lanes[selectable_lanes[0]].front_drivable_length > OBSERVE_RANGE - LANE_CHANGE_BASE_LENGTH:
                # 添加  or lane_drivable_length[0] > OBSERVE_RANGE - LANE_CHANGE_BASE_LENGTH是为了避免视野内障碍物还没看全就急于做出变道决策，要保证有变道的足够空间。
                target_lane_id = cur_lane_info.cur_lane_id
            else:
                if lane_reward[1] > lane_reward[0] * 1.2 + EPS and lane_reward[2] > lane_reward[0] * 1.2 + EPS:
                    if lane_reward[2] > lane_reward[1] * 1.2 + EPS and lane_drivable_length[2] > EPS + LANE_CHANGE_BASE_LENGTH:
                        target_lane_id = cur_lane_info.right_lane_id
                    elif lane_reward[2] <= lane_reward[1] * 1.2 + EPS and lane_drivable_length[1] > EPS + LANE_CHANGE_BASE_LENGTH:
                        target_lane_id = cur_lane_info.left_lane_id
                elif lane_reward[1] > lane_reward[0] * 1.2 + EPS and lane_drivable_length[1] > EPS + LANE_CHANGE_BASE_LENGTH:
                    target_lane_id = cur_lane_info.left_lane_id
                elif lane_reward[2] > lane_reward[0] * 1.2 + EPS and lane_drivable_length[2] > EPS + LANE_CHANGE_BASE_LENGTH:
                    target_lane_id = cur_lane_info.right_lane_id

                if target_lane_id == -1:
                    if lane_drivable_length[1] - lane_drivable_length[0] > EPS + 2 * LANE_CHANGE_BASE_LENGTH and lane_reward[1] - lane_reward[0] > EPS:
                        target_lane_id = cur_lane_info.left_lane_id
                    elif lane_drivable_length[2] - lane_drivable_length[0] > EPS + 2 * LANE_CHANGE_BASE_LENGTH and lane_reward[2] - lane_reward[0] > EPS:
                        target_lane_id = cur_lane_info.right_lane_id
                    else:
                        target_lane_id = cur_lane_info.cur_lane_id
        if target_lane_id <= 0:
            target_lane_id = cur_lane_info.cur_lane_id
        if target_lane_id == cur_lane_info.left_lane_id and can_change_constraint[1] <= 0:
            target_lane_id = cur_lane_info.cur_lane_id
        if target_lane_id == cur_lane_info.right_lane_id and can_change_constraint[2] <= 0:
            target_lane_id = cur_lane_info.cur_lane_id

    # elif scenario == "intersection":
    #     target_lane_id = cur_lane_info.cur_lane_id

    else:
        target_lane_id = cur_lane_info.cur_lane_id

    # 如果路走完了,选择下一条目标车道
    # 选择标准：先选优先级高的，同优先级的车道里选择可行长度大的。
    next_lane_id = -1
    speed_upper_limit = SPEED_UPPER_LIMIT_DEFAULT
    if cur_lane_info.cur_lane_id > 0:
        speed_upper_limit = min(speed_upper_limit, (lane_list[cur_lane_info.cur_lane_id].speedUpperLimit / 3.6))
    desired_length = max(2 * LANE_CHANGE_BASE_LENGTH, speed_upper_limit * 3)

    if target_lane_id != -1:
        if available_lanes[target_lane_id].after_length < desired_length:
            rospy.loginfo("need to connect next road")
            rospy.loginfo("available lead to ids %s " % list(lane_list[target_lane_id].leadToIds))
            next_lane_id = -1
            temp_drivable_length = 0
            next_lane_found = 0
            temp_priority_max = 0
            for index in lane_list[target_lane_id].leadToIds:
                if lane_list[index].priority == 2 and available_lanes[index].front_drivable_length > temp_drivable_length:
                    next_lane_id = index
                    temp_drivable_length = available_lanes[index].front_drivable_length
                    next_lane_found = 1
            if next_lane_found == 0:
                for index in lane_list[target_lane_id].leadToIds:
                    if lane_list[index].priority == 1 and available_lanes[
                        index].front_drivable_length > temp_drivable_length:
                        next_lane_id = index
                        temp_drivable_length = available_lanes[index].front_drivable_length
                        next_lane_found = 1
            if next_lane_found == 0:
                for index in lane_list[target_lane_id].leadToIds:
                    if lane_list[index].priority > temp_priority_max:
                        temp_priority_max = lane_list[index].priority
                        next_lane_id = index
                        next_lane_found = 1

    rospy.loginfo("this target lane id %d" % target_lane_id)
    rospy.loginfo("next target lane id%d" % next_lane_id)
    return target_lane_id, next_lane_id


def lanes_of_interest_selector(lane_list, pose_data, scenario, available_lanes, target_lane_id, next_lane_id):
    """
    :param lane_list:
    :param pose_data:
    :return:
    """
    lanes_of_interest = {}
    # for merge and across scenario
    if scenario == "merge":
        speed_limit = []
        lane_id = []
        lateral_dis_to_target_lane = []
        project_x, project_y = [], []
        project_s, project_l = [], []
        vehicle_dis_to_target_lane = available_lanes[target_lane_id].lateral_distance

        for lane_index in available_lanes.keys():
            lane_id.append(lane_index)
            speed_limit.append(lane_list[lane_index].speedUpperLimit / 3.6)
            project_x.append(available_lanes[lane_index].projection_x)
            project_y.append(available_lanes[lane_index].projection_y)
            project_s.append(available_lanes[lane_index].before_length)
            project_l.append(available_lanes[lane_index].lateral_distance)

        # 将各车道的投影点投影到目标车道，并记录投影横向距离
        target_lane = lane_list[target_lane_id]
        points_x, points_y = [], []
        for j in range(len(target_lane.points)):
            points_x.append(target_lane.points[j].x)
            points_y.append(target_lane.points[j].y)
        points_num = len(points_x)
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        for i in range(len(lane_id)):
            projection_result = lane_projection(points_x, points_y, points_num,
                                                project_x[i], project_y[i])
            lateral_dis_to_target_lane.append(projection_result[3])

        # 选择横向距离在vehicle_dis_to_target_lane到 0 之间的车道作为感兴趣车道
        # (lateral_dis_to_target_lane-vehicle_dis_to_target_lane)*(0-lateral_dis_to_target_lane)>0
        if vehicle_dis_to_target_lane < 0:
            for i in range(len(lane_id)):
                if lateral_dis_to_target_lane[i] > vehicle_dis_to_target_lane and lateral_dis_to_target_lane[i] < 0.1:
                    lane_of_interest = LanesOfInterest()
                    lane_of_interest.lane_id = lane_id[i]
                    lane_of_interest.end_s = project_s[i]
                    time_required = TIME_ACC + TIME_DELAY + TIME_SPACE + abs(project_l[i]) / 5 * 3.6
                    lane_of_interest.action_time = time_required
                    longitudinal_distance = speed_limit[i] / 3.6 * time_required
                    if project_s[i] < longitudinal_distance:
                        lane_of_interest.start_s = 0
                    else:
                        lane_of_interest.start_s = project_s[i] - longitudinal_distance
                    lanes_of_interest[lane_id[i]] = lane_of_interest
        else:
            for i in range(len(lane_id)):
                if lateral_dis_to_target_lane[i] < vehicle_dis_to_target_lane and lateral_dis_to_target_lane[i] > -0.1:
                    lane_of_interest = LanesOfInterest()
                    lane_of_interest.lane_id = lane_id[i]
                    lane_of_interest.end_s = project_s[i]
                    time_required = TIME_ACC + TIME_DELAY + TIME_SPACE + abs(project_l[i]) / 5 * 3.6
                    lane_of_interest.action_time = time_required
                    longitudinal_distance = speed_limit[i] / 3.6 * time_required
                    if project_s[i] < longitudinal_distance:
                        lane_of_interest.start_s = 0
                    else:
                        lane_of_interest.start_s = project_s[i] - longitudinal_distance
                    lanes_of_interest[lane_id[i]] = lane_of_interest

    elif scenario == "intersection":
        # 判断当前目标车道的下一条车道是否是别的车道的下一条车道，如果是的话，考虑汇入的情况
        interest_lanes_found = 0
        lead_to_id = next_lane_id
        for index in lane_list.keys():
            if lead_to_id in lane_list[index].leadToIds:
                lane_of_interest = LanesOfInterest()
                lane_of_interest.lane_id = index
                lane_of_interest.end_s = available_lanes[index].before_length + available_lanes[index].after_length
                time_required = TIME_ACC + TIME_DELAY + TIME_SPACE + available_lanes[target_lane_id].after_length / max(
                    pose_data.mVf, 5 / 3.6)
                lane_of_interest.action_time = time_required
                longitudinal_distance = (lane_list[index].speedUpperLimit / 3.6) * time_required
                if lane_of_interest.end_s < longitudinal_distance:
                    lane_of_interest.start_s = 0
                else:
                    lane_of_interest.start_s = lane_of_interest.end_s - longitudinal_distance
                lanes_of_interest[index] = lane_of_interest
                interest_lanes_found = 1
        if interest_lanes_found == 0 and next_lane_id != -1:
            # 考虑下一个连接点是否有汇入情况
            if len(lane_list[next_lane_id].leadToIds) > 0:
                lead_to_id = lane_list[next_lane_id].leadToIds[0]
                for index in lane_list.keys():
                    if lead_to_id in lane_list[index].leadToIds:
                        lane_of_interest = LanesOfInterest()
                        lane_of_interest.lane_id = index
                        lane_of_interest.end_s = available_lanes[index].before_length + available_lanes[
                            index].after_length
                        time_required = TIME_ACC + TIME_DELAY + TIME_SPACE + (
                                    available_lanes[target_lane_id].after_length + available_lanes[
                                next_lane_id].after_length) / max(pose_data.mVf, 5 / 3.6)
                        lane_of_interest.action_time = time_required
                        longitudinal_distance = (lane_list[index].speedUpperLimit / 3.6) * time_required
                        if lane_of_interest.end_s < longitudinal_distance:
                            lane_of_interest.start_s = 0
                        else:
                            lane_of_interest.start_s = lane_of_interest.end_s - longitudinal_distance
                        lanes_of_interest[index] = lane_of_interest
    rospy.loginfo("lanes of interest ids %s in scenario%s" % (lanes_of_interest.keys(), scenario))
    return lanes_of_interest


def initial_priority_decider(lanes_of_interest, obstacles_list):
    """
    在开始 merge 或者 across 之前，确保每条感兴趣车道的感兴趣区域内的动态障碍物通行时间大于我的行动时间
    :param lanes_of_interest:
    :param obstacles_list:
    :return:
    """
    is_ready = True
    for lane_index in lanes_of_interest.keys():
        loi_info = lanes_of_interest[lane_index]
        min_time = 1000
        action_time = loi_info.action_time
        for obstacle_index in obstacles_list.keys():
            obstacle_info = obstacles_list[obstacle_index]
            if obstacle_info.cur_lane_id == lane_index:
                if loi_info.end_s > obstacle_info.s_record[-1] > loi_info.start_s:
                    if obstacle_info.s_velocity[-1] > 0:
                        temp_time = (loi_info.end_s - obstacle_info.s_record[-1]) / obstacle_info.s_velocity[-1]
                        rospy.loginfo("obstacle %d coming in %f seconds" % (obstacle_index, temp_time))
                        if temp_time < min_time:
                            min_time = temp_time
                        if temp_time < action_time:
                            # 其余属于感兴趣区域内的同向动态障碍物，小决策置超车
                            obstacle_info.sub_decision = GIVE_WAY
                            obstacle_info.safe_distance = MIN_SAFE_DISTANCE
                        else:
                            obstacle_info.sub_decision = OVERTAKE
                            obstacle_info.safe_distance = MIN_SAFE_DISTANCE

        if min_time < action_time:
            is_ready = False
    return is_ready


def merge_priority_decider(target_lane_id, obstacles_list, pose_data, available_lanes):
    """
    当已经开始执行 merge， 考虑在动态环境中的优先级，运动中的 merge，考虑超车
    :param target_lane_id:
    :param obstacles_list:
    :return:
    """
    speed_upper_limit = SPEED_UPPER_LIMIT_DEFAULT
    # is_ready = True

    vehicle_s = available_lanes[target_lane_id].before_length

    # target_lane_obstacles_id = []
    # target_lane_obstacles_s = []
    target_lane_obstacles_s_id = []
    for obstacle_index in obstacles_list.keys():
        obstacle_info = obstacles_list[obstacle_index]
        if obstacle_info.cur_lane_id == target_lane_id:
            # target_lane_obstacles_id.append(obstacle_index)
            # target_lane_obstacles_s.append(obstacle_info.s_record[-1])
            target_lane_obstacles_s_id.append([obstacle_info.s_record[-1], obstacle_index])

    target_lane_obstacles_s_id.sort()
    # # insert sort
    # count = len(target_lane_obstacles_id)
    # for i in range(1, count):
    #     key = target_lane_obstacles_s[i]
    #     id = target_lane_obstacles_id[i]
    #     j = i - 1
    #     while (j >= 0):
    #         if target_lane_obstacles_s[j] > key:
    #             target_lane_obstacles_s[j + 1] = target_lane_obstacles_s[j]
    #             target_lane_obstacles_s[j] = key
    #             target_lane_obstacles_id[j + 1] = target_lane_obstacles_id[j]
    #             target_lane_obstacles_id[j] = id
    #         j -= 1

    target_lane_obstacles_s_id.append([vehicle_s + OBSERVE_RANGE, -10])
    # 在最前方障碍物前面加上一个辅助动态障碍物
    target_slot = -1
    # target_slot采用正常计数规则，
    # 0表示在第一辆车之前插入，
    # i表示在第i辆车与第i+1辆车之间插入index=[i-1,i]
    # i+1表示在第i+1辆车之后插入
    # 每次循环都要给target_slot更新值，如果确定找到则是i+1,否则是先赋值i，再往前找。
    # 找到的目标slot为[target_slot-1,target_slot]
    for i in range(len(target_lane_obstacles_s_id) - 2, -1, -1):
        # 0 到 len(target_lane_obstacles_s_id)-2 为感知到的障碍物
        # 1) the slot’s front - obstacle back - merge and rear - obstacle front - merge are feasible;
        # 2) the gap between obstacles is large enough for Boss plus proper spacing in front and rear;
        # 3) the front obstacle’s velocity is greater than or equal to the rear obstacle’s velocity, so the gap is not closing;
        # 4) the merge-between point will be reached before the checkpoint.
        if target_lane_obstacles_s_id[i][0] > vehicle_s:
            target_slot = i
            continue
        else:
            # slot[i,i+1]
            if i == len(target_lane_obstacles_s_id) - 2:
                rear_obstacle = obstacles_list[target_lane_obstacles_s_id[i][1]]
                if target_lane_obstacles_s_id[i+1][0] - target_lane_obstacles_s_id[i][0] > 2 * (
                        desired_safety_distance(rear_obstacle.s_velocity[-1]) + VEHICLE_LENGTH):
                    target_slot = i + 1
                    break
                else:
                    target_slot = i
                    continue
            else:
                front_obstacle = obstacles_list[target_lane_obstacles_s_id[i + 1][1]]
                rear_obstacle = obstacles_list[target_lane_obstacles_s_id[i][1]]
                if target_lane_obstacles_s_id[i+1][0] - target_lane_obstacles_s_id[i][0] > 2 * (
                        desired_safety_distance(rear_obstacle.s_velocity[-1]) + VEHICLE_LENGTH) and \
                        1.2 * front_obstacle.s_velocity[-1] > rear_obstacle.s_velocity[-1]:
                    target_slot = i + 1
                    break
                else:
                    target_slot = i
                    continue

    rospy.loginfo("target_lane_obstacles_s_id %s" % target_lane_obstacles_s_id)
    rospy.loginfo("vehicle s %f" % vehicle_s)
    rospy.loginfo("target insert slot %d" % (target_slot))

    if target_slot != -1:
        if target_slot == 0:
            if target_lane_obstacles_s_id[target_slot][0] > vehicle_s + desired_safety_distance(pose_data.mVf):
                is_ready = True
            else:
                speed_upper_limit = obstacles_list[target_lane_obstacles_s_id[target_slot][1]].s_velocity[-1] * 0.8
                is_ready = False
        else:
            if target_lane_obstacles_s_id[target_slot][0] > vehicle_s + desired_safety_distance(pose_data.mVf) and \
                    target_lane_obstacles_s_id[target_slot - 1][0] < vehicle_s - MIN_GAP_DISTANCE:
                is_ready = True
            else:
                if target_slot == len(target_lane_obstacles_s_id) - 1:
                    speed_upper_limit = SPEED_UPPER_LIMIT_DEFAULT
                else:
                    speed_upper_limit = obstacles_list[target_lane_obstacles_s_id[target_slot][1]].s_velocity[-1] * 0.8
                is_ready = False
    else:
        # 没有动态障碍物
        is_ready = True
    rospy.loginfo("is ready to lane change? %d" % is_ready)
    rospy.loginfo("set speed upper limit to a certain value %f based on other obstacle's velocity" % speed_upper_limit)
    return is_ready, speed_upper_limit


def obstacle_of_interest_selector(obstacles_list, available_lanes = {}, current_lane_id = -1, target_lane_id = -1, next_target_lane_id = -1):
    if target_lane_id != -1:
        closest_regular_object_id = available_lanes[target_lane_id].closest_regular_object_id
        if closest_regular_object_id > 0:
            rospy.loginfo("give way to obstacle id %d on lane %d distance %f" % (closest_regular_object_id, target_lane_id, (obstacles_list[closest_regular_object_id].s_record[-1] - available_lanes[target_lane_id].before_length)))
            obstacles_list[closest_regular_object_id].sub_decision = GIVE_WAY
            obstacles_list[closest_regular_object_id].safe_distance = MIN_GAP_DISTANCE
    if next_target_lane_id != -1:
        closest_regular_object_id = available_lanes[next_target_lane_id].closest_regular_object_id
        if closest_regular_object_id > 0:
            rospy.loginfo("give way to obstacle id %d on lane %d distance %f" % (
            closest_regular_object_id, target_lane_id,
            (obstacles_list[closest_regular_object_id].s_record[-1] - available_lanes[target_lane_id].before_length)))
            obstacles_list[closest_regular_object_id].sub_decision = GIVE_WAY
            obstacles_list[closest_regular_object_id].safe_distance = MIN_GAP_DISTANCE


def speed_limit_decider(lane_list, available_lanes, current_lane_info, target_lane_id, pose_data, action_decelerate=-1, merge_decelerate=-1, certain_speed_limit = -1):
    # speed_upper_limit_default = 30
    # speed_lower_limit_default = 0
    speed_upper_limit = SPEED_UPPER_LIMIT_DEFAULT
    speed_lower_limit = SPEED_LOWER_LIMIT_DEFAULT
    if current_lane_info.cur_lane_id > 0:
        speed_upper_limit = min(speed_upper_limit, (lane_list[current_lane_info.cur_lane_id].speedUpperLimit / 3.6))
        speed_lower_limit = max(speed_lower_limit, (lane_list[current_lane_info.cur_lane_id].speedLowerLimit / 3.6))
        if available_lanes[current_lane_info.cur_lane_id].front_drivable_length < pose_data.mVf ** 2 / (2 * COMFORT_DEC):
            speed_upper_limit = min(speed_upper_limit, 1.5)
    if target_lane_id > 0:
        speed_upper_limit = min(speed_upper_limit, (lane_list[target_lane_id].speedUpperLimit / 3.6))
        speed_lower_limit = max(speed_lower_limit, (lane_list[target_lane_id].speedLowerLimit / 3.6))
        if available_lanes[target_lane_id].front_drivable_length < pose_data.mVf ** 2 / (2 * COMFORT_DEC):
            speed_upper_limit = min(speed_upper_limit, 1.5)
    if merge_decelerate != -1:
        speed_upper_limit = min(SPEED_UPPER_LIMIT_MERGE, merge_decelerate / 5 * speed_upper_limit)
        # 分5个减速等级

    if action_decelerate != -1:
        # 分5个减速等级
        speed_upper_limit = action_decelerate / 5 * speed_upper_limit

    if certain_speed_limit != -1:
        speed_upper_limit = min(speed_upper_limit, certain_speed_limit)

    rospy.loginfo("speed upper %f" % speed_upper_limit)
    rospy.loginfo("speed lower %f" % speed_lower_limit)
    return speed_upper_limit, speed_lower_limit


def history_lanes_recorder(current_lane_id):
    global history_lane_ids
    if history_lane_ids == []:
        history_lane_ids.append(current_lane_id)
    else:
        if history_lane_ids[-1] != current_lane_id:
            history_lane_ids.append(current_lane_id)
    rospy.loginfo("the lanes that has passed through %s " % list(history_lane_ids))


def desired_length_decider(available_lanes, target_lane_id, speed_upper_limit, scenario = 'lane_follow'):
    desired_length = 0
    if target_lane_id != -1:
        desired_length = max(2 * LANE_CHANGE_BASE_LENGTH, speed_upper_limit * 3)
        # if there is a static obstacle in the way
#        if min(available_lanes[target_lane_id].after_length, OBSERVE_RANGE) - available_lanes[
#             target_lane_id].front_drivable_length > EPS:
#             # if (current_lane_info.can_change_left or current_lane_info.can_change_right) and available_lanes[
#             #     target_lane_id].closest_moving_object_type != 'VEHICLE':
#             # it the static obstacle ahead is a vehicle, then stop behind the obstacle. otherwise, try to reach as far as possible.
#            if available_lanes[target_lane_id].closest_static_object_type != 'VEHICLE':
#                pass
#            else:
#                 desired_length = min(desired_length,
#                                      available_lanes[target_lane_id].front_drivable_length - MIN_GAP_DISTANCE)

        # if the target lane is beyond search, stop providing reference path.
        if scenario == 'lane_follow':
            if abs(available_lanes[target_lane_id].lateral_distance) > 10 or abs(
                    available_lanes[target_lane_id].dir_diff_signed > math.pi / 2):
                desired_length = 0

        rospy.loginfo("drivable length %f" % available_lanes[target_lane_id].front_drivable_length)
        rospy.loginfo("desired length %f" % desired_length)
    return desired_length


def points_filler(lane_list, target_lane_id, next_lane_id, available_lanes, target_length):
    """
    如果填充到最后没有达到目标长度，就终止填充
    :param lane_list:
    :param target_lane_id:
    :param next_lane_id:
    :param available_lanes:
    :param target_length:
    :return:
    """
    ref_path = []
    temp_path_length = 0
    if target_lane_id != -1:
        target_lane_num = len(lane_list[target_lane_id].points)
        target_lane_index = available_lanes[target_lane_id].projection_index

        if next_lane_id != -1:
            next_lane_num = len(lane_list[next_lane_id].points)
            # next_lane_index = available_lanes[next_lane_id].projection_index
            next_lane_index = 0
            # 下一条路从头开始填充
        else:
            next_lane_num, next_lane_index = 0, 0

        # 取第一段
        ref_path.append(lane_list[target_lane_id].points[target_lane_index])
        temp_seg_vec = np.array(
            [lane_list[target_lane_id].points[target_lane_index + 1].x - available_lanes[target_lane_id].projection_x,
             lane_list[target_lane_id].points[target_lane_index + 1].y - available_lanes[target_lane_id].projection_y])
        temp_seg_length = np.linalg.norm(temp_seg_vec)
        if temp_seg_length > target_length:
            ratio = target_length / temp_seg_length
            x = available_lanes[target_lane_id].projection_x + ratio * temp_seg_vec[0]
            y = available_lanes[target_lane_id].projection_y + ratio * temp_seg_vec[1]
            temp_point = Point32()
            temp_point.x = x
            temp_point.y = y
            temp_point.z = 0
            ref_path.append(temp_point)
            temp_path_length += target_length
        else:
            temp_path_length += temp_seg_length
            target_lane_index += 1
            ref_path.append(lane_list[target_lane_id].points[target_lane_index])

        # 第二段开始
        while (temp_path_length < target_length):
            if target_lane_index < target_lane_num - 1:
                # 从当前目标车道上选点
                temp_seg_vec = np.array([lane_list[target_lane_id].points[target_lane_index + 1].x -
                                         lane_list[target_lane_id].points[target_lane_index].x,
                                         lane_list[target_lane_id].points[target_lane_index + 1].y -
                                         lane_list[target_lane_id].points[target_lane_index].y])
                temp_seg_length = np.linalg.norm(temp_seg_vec)
                target_seg_length = target_length - temp_path_length
                if temp_seg_length > target_seg_length:
                    ratio = target_seg_length / temp_seg_length
                    x = lane_list[target_lane_id].points[target_lane_index].x + ratio * temp_seg_vec[0]
                    y = lane_list[target_lane_id].points[target_lane_index].y + ratio * temp_seg_vec[1]
                    temp_point = Point32()
                    temp_point.x = x
                    temp_point.y = y
                    temp_point.z = 0
                    ref_path.append(temp_point)
                    temp_path_length += target_seg_length
                    break
                else:
                    temp_path_length += temp_seg_length
                    target_lane_index += 1
                    ref_path.append(lane_list[target_lane_id].points[target_lane_index])
            elif next_lane_index < next_lane_num - 1 and next_lane_num != 0:
                # 从下一条目标车道上选点
                # print(target_length, temp_path_length)
                temp_seg_vec = np.array([lane_list[next_lane_id].points[next_lane_index + 1].x -
                                         lane_list[next_lane_id].points[next_lane_index].x,
                                         lane_list[next_lane_id].points[next_lane_index + 1].y -
                                         lane_list[next_lane_id].points[next_lane_index].y])
                temp_seg_length = np.linalg.norm(temp_seg_vec)
                target_seg_length = target_length - temp_path_length
                if temp_seg_length > target_seg_length:
                    ratio = target_seg_length / temp_seg_length
                    x = lane_list[next_lane_id].points[next_lane_index].x + ratio * temp_seg_vec[0]
                    y = lane_list[next_lane_id].points[next_lane_index].y + ratio * temp_seg_vec[1]
                    temp_point = Point32()
                    temp_point.x = x
                    temp_point.y = y
                    temp_point.z = 0
                    ref_path.append(temp_point)
                    temp_path_length += target_seg_length
                    break
                else:
                    temp_path_length += temp_seg_length
                    next_lane_index += 1
                    ref_path.append(lane_list[next_lane_id].points[next_lane_index])
            else:
                # 无下一条车道，或者下一条车道点已经取完
                break

    rospy.loginfo("filled path length %f" % temp_path_length)
    return ref_path


def desired_safety_distance(velocity):
    return max(1, velocity * 3.6 / 10) * MIN_GAP_DISTANCE


def output_filler(scenario=0, filtered_obstacles={}, speed_upper_limit=0, speed_lower_limit=0, reference_path=[], selected_parking_lot=[], light_detection_switch = 0):
    message = Decision()
    if scenario_error_handle == NONE:
        message.scenario = int(scenario_error_handle)
    else:
        message.scenario = int(scenario)
        message.speedUpperLimit = speed_upper_limit
        message.speedLowerLimit = speed_lower_limit
        message.refPath = reference_path
        message.selectedParkingLot = selected_parking_lot
        message.lightDetectionSwitch = light_detection_switch
        filtered_obstacles_list = []
        for obstacle_id, obstacle in filtered_obstacles.items():
            filtered_obstacle = FilteredObstacle()
            filtered_obstacle.type = obstacle.type
            filtered_obstacle.width = obstacle.width
            filtered_obstacle.length = obstacle.length
            filtered_obstacle.velocity = obstacle.cur_velocity_vec
            filtered_obstacle.points = obstacle.cur_bounding_points
            filtered_obstacle.isMoving = obstacle.is_moving
            filtered_obstacle.decision = obstacle.sub_decision
            filtered_obstacle.safeDistance = obstacle.safe_distance
            filtered_obstacle.predictedCenterPointsTrajectory = obstacle.predicted_center_points
            filtered_obstacle.predictedHeadings = obstacle.predicted_headings
            # for point in obstacle.predicted_center_points:
            #     temp_point = Point32()
            #     temp_point.x = point[0]
            #     temp_point.y = point[1]
            #     temp_point.z = 0
            #     filtered_obstacle.predictedCenterPointsTrajectory.append(temp_point)
            # for heading in obstacle.predicted_headings:
            #     filtered_obstacle.predictedHeadings.append(heading)
            filtered_obstacles_list.append(filtered_obstacle)
        message.filteredObstacles = filtered_obstacles_list

    aa = rospy.get_time()
    global bb
    rospy.loginfo("decision processing duration %f" % (aa - bb))
    bb = aa

    rospy.loginfo("output message updated at %s" % rospy.get_time())
    decision_msg_pub.publish(message)


def re_global_planning_caller(blocked_id_list = []):
    try:
        rospy.loginfo("re-global planning because of lane %s blocked." % list(blocked_id_list))
        map_provider_respond = re_global_planning(blocked_id_list)
        if map_provider_respond.received == 1:
            rospy.sleep(1)
            return
    except rospy.ServiceException as exception:
        rospy.loginfo("Service did not process request: " + str(exception))


def mission_finished_caller():
    try:
        global mission_ahead
        rospy.loginfo("mission index %d completed." % mission_ahead.missionIndex)
        mission_planning_respond = current_mission_finished(mission_ahead.missionIndex)
        if mission_planning_respond.received == 1:
            mission_ahead = MissionAhead()
            return
    except rospy.ServiceException as exception:
        rospy.loginfo("Service did not process request: " + str(exception))


# def re_global_planning():
#     rospy.init_node('greetings_client')
#     #   等待有可用的服务"greetings"
#     rospy.wait_for_service("greetings")
#     try:
#         # 定义service客户端，service 名称为 “greetings”，service 类型为 Greeting
#         greetings_client = rospy.ServiceProxy("greetings", Greeting)
#         # 向server端发送请求,发送的request内容为 name 和 age，其值分别为 "HAN", 20
#         # 此处发送的 request 内容与 srv 文件中定义的 request 部分的属性是一致的
#         # resp = greetings_client("HAN",20)
#         resp = greetings_client.call("HAN", 20)
#         rospy.loginfo("Message From server:%s" % resp.feedback)
#     except rospy.ServiceException as e:
#         rospy.logwarn("Service call failed: %s" % e)


#########################################
# STARTUP
#########################################
class StartupCheck(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['ready'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list'])

    def execute(self, user_data):
        # clear the global variables
        global parking_lane_id, target_parking_slot, target_parking_slot_projection
        parking_lane_id = 0
        target_parking_slot = []
        target_parking_slot_projection = []
        rospy.sleep(4 * DECISION_PERIOD)
        # reset the output
        while not rospy.is_shutdown():
            rospy.loginfo("currently in StartupCheck")

            rospy.sleep(DECISION_PERIOD)

            if mission_ahead.missionType is None:
                continue

            if scenario_error_handle != REF_PATH_FOLLOW:
                output_filler(scenario=NONE)
                continue

            user_data_updater(user_data)

            # check every input
            if user_data.obstacles_list == {}:
                rospy.loginfo("obstacle message missing.")
            if user_data.lights_list == {}:
                rospy.loginfo("light message missing.")
            if user_data.signs_data == {}:
                rospy.loginfo("sign message missing.")

            if user_data.lane_list == {} or user_data.pose_data is None:
                rospy.loginfo("nothing from input.")
                continue
            # check vehicle status
            vehicle_status = 1
            if vehicle_status == 0:
                continue
            return 'ready'


class Startup(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['in_lane_driving', 'merge_and_across', 'park'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in Startup")
            output_filler(scenario=NONE)
            user_data_updater(user_data)
            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            rospy.loginfo("mission type: %s on lane: %s " % (mission_ahead.missionType, list(mission_ahead.missionLaneIds)))
            # 执行到当前mission的最后一段
            if current_lane_info.cur_lane_id != -1:
                if history_lane_ids[-1] in mission_ahead.missionLaneIds:
                    if mission_ahead.missionType == 'park':
                        if mission_ahead.missionThingId in parking_area_list.keys():
                            rospy.loginfo("park into a parking lot")
                            sum_angle = 0
                            global target_parking_area
                            target_parking_area = parking_area_list[mission_ahead.missionThingId]
                            for i in range(len(target_parking_area) - 1):
                                vehicle_to_point1 = np.array([target_parking_area[i].x - user_data.pose_data.mapX,
                                                              target_parking_area[i].y - user_data.pose_data.mapY])
                                vehicle_to_point2 = np.array([target_parking_area[i + 1].x - user_data.pose_data.mapX,
                                                              target_parking_area[i + 1].y - user_data.pose_data.mapY])
                                length1 = np.linalg.norm(vehicle_to_point1)
                                length2 = np.linalg.norm(vehicle_to_point2)
                                cos_value = np.dot(vehicle_to_point1, vehicle_to_point2) / (length1 * length2)
                                if cos_value > 1:
                                    cos_value = 1
                                if cos_value < -1:
                                    cos_value = -1
                                sum_angle += math.acos(cos_value)
                            if abs(sum_angle - 2 * math.pi) < EPS:
                                return 'park'
                        # 目标是固定的车位
                        if mission_ahead.missionThingId in parking_slots_list.keys():
                            rospy.loginfo("park into a certain parking slot")
                            temp_parking_slot = parking_slots_list[mission_ahead.missionThingId].points
                            for i in range(len(temp_parking_slot)):
                                if math.sqrt(math.pow(user_data.pose_data.mapX - temp_parking_slot[i].x, 2) + math.pow(
                                        user_data.pose_data.mapY - temp_parking_slot[i].y, 2)) < 30:
                                    return 'park'

            if current_lane_info.cur_lane_id == -1 or current_lane_info.cur_priority <= 0:
                # 的找不到当前车道或者当前车道优先级小，进入merge
                return 'merge_and_across'
            else:
                return 'in_lane_driving'
            # 还应该判断是否周围能找到目标车道，如果找不到并且周围有引导，应该进入非结构化行驶。


#########################################
# IN LANE DRIVING
#########################################
class InLaneDriving(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['park', 'intersection', 'merge_and_across', 'need_to_change_lane', 'error', 'finished', 'stop_line'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        """
        在不满足正常变道的条件下，必须停下而不是试探性往前开的情况：
        1. 停止线前停车
        2. 没有变道的条件
        3. 前方堵住路的事静态的车辆
        """
        while not rospy.is_shutdown():
            rospy.loginfo("currently in InLaneDriving")
            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)
            blocked_lane_id_list = []
            global planning_feedback, last_target_lane_id_ugv, last_chosen_lane_id_ugv, lane_change_consideration_start_time

            if scenario_error_handle != REF_PATH_FOLLOW:
                output_filler(scenario=NONE)
                return 'finished'
#            if planning_feedback == RE_CHOOSE_PATH:
#                planning_feedback = 0
#                rospy.loginfo("No way to go!!!")
#                blocked_lane_id_list.append(current_lane_info.cur_lane_id)
#                rospy.loginfo("blocked way ids %s " % list(blocked_lane_id_list))
#                re_global_planning_caller(blocked_lane_id_list)
#                output_filler(scenario=NONE)
#                return 'finished'

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            if current_lane_info.cur_lane_id == -1 or current_lane_info.cur_priority <= 0:
                # 的找不到当前车道或者当前车道优先级小，进入merge
                return 'merge_and_across'

            if current_lane_info.next_stop_type == OBSERVE_TRAFFIC_LIGHT and current_lane_info.dist_to_next_stop < max(user_data.pose_data.mVf**2 / 2 / COMFORT_DEC, 30):
                return 'intersection'

            if current_lane_info.next_stop_type == STOP_GIVE_WAY and current_lane_info.dist_to_next_stop < max(user_data.pose_data.mVf**2 / 2 / COMFORT_DEC, 30):
                return 'stop_line'

            rospy.loginfo("mission type: %s on lane: %s " % (mission_ahead.missionType, list(mission_ahead.missionLaneIds)))
            # 执行到当前mission的最后一段
            if current_lane_info.cur_lane_id != -1:
                if history_lane_ids[-1] in mission_ahead.missionLaneIds:
                    if mission_ahead.missionType == 'road':
                        mission_finished_caller()
                        output_filler(scenario=NONE)
                        return 'finished'
                    if mission_ahead.missionType == 'park':
                        if mission_ahead.missionThingId in parking_area_list.keys():
                            rospy.loginfo("park into a parking lot")
                            sum_angle = 0
                            global target_parking_area
                            target_parking_area = parking_area_list[mission_ahead.missionThingId]
                            for i in range(len(target_parking_area) - 1):
                                vehicle_to_point1 = np.array([target_parking_area[i].x - user_data.pose_data.mapX,
                                                              target_parking_area[i].y - user_data.pose_data.mapY])
                                vehicle_to_point2 = np.array([target_parking_area[i + 1].x - user_data.pose_data.mapX,
                                                              target_parking_area[i + 1].y - user_data.pose_data.mapY])
                                length1 = np.linalg.norm(vehicle_to_point1)
                                length2 = np.linalg.norm(vehicle_to_point2)
                                cos_value = np.dot(vehicle_to_point1, vehicle_to_point2) / (length1 * length2)
                                if cos_value > 1:
                                    cos_value = 1
                                if cos_value < -1:
                                    cos_value = -1
                                sum_angle += math.acos(cos_value)
                            if abs(sum_angle - 2 * math.pi) < EPS:
                                return 'park'
                        # 目标是固定的车位
                        if mission_ahead.missionThingId in parking_slots_list.keys():
                            rospy.loginfo("park into a certain parking slot")
                            temp_parking_slot = parking_slots_list[mission_ahead.missionThingId].points
                            for i in range(len(temp_parking_slot)):
                                if math.sqrt(math.pow(user_data.pose_data.mapX - temp_parking_slot[i].x, 2) + math.pow(
                                        user_data.pose_data.mapY - temp_parking_slot[i].y, 2)) < 30:
                                    return 'park'

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)
            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data, 'lane_follow',
                                                                current_lane_info, available_lanes)

            this_target_lane_id_ugv = target_lane_id

            if last_chosen_lane_id_ugv == this_target_lane_id_ugv:
                target_lane_id = last_chosen_lane_id_ugv
            else:
                target_lane_id = current_lane_info.cur_lane_id

            rospy.loginfo("final target lane id %d" % target_lane_id)
            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)

            # 避免找到身后的目标路径
            if desired_length > 0:
                reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id, available_lanes,
                                                   desired_length)
            else:
                reference_path = []

            obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id, target_lane_id, next_lane_id)

            # if the vehicle on the surrounding lanes is about to cut into this lane. decelerate.
            output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit, reference_path)

            last_chosen_lane_id_ugv = target_lane_id

            if this_target_lane_id_ugv != current_lane_info.cur_lane_id:
                last_target_lane_id_ugv = this_target_lane_id_ugv
                lane_change_consideration_start_time = rospy.get_time()
                return 'need_to_change_lane'

            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


class LaneChangePreparing(smach.State):
    """
    enter this state when there is a need to change lane.
    select the target lane considering mission and obstacles.
    then, choose a target interval on the target lane.
    return to main state when there is no need to change lane.
    """

    def __init__(self):
        smach.State.__init__(self, outcomes=['cancel_intention', 'ready_to_change_lane'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        global lane_change_consideration_start_time, lane_change_calm_down_start_time, last_chosen_lane_id_ugv
        while not rospy.is_shutdown():
            rospy.loginfo("currently in LaneChangePreparing")

            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            if current_lane_info.cur_lane_id == -1 or current_lane_info.cur_priority <= 0:
                # 的找不到当前车道或者当前车道优先级小，进入merge
                lane_change_consideration_start_time = 0
                return 'cancel_intention'

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)

            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data, 'lane_follow',
                                                                current_lane_info, available_lanes)

            this_target_lane_id_ugv = target_lane_id
            if last_chosen_lane_id_ugv == this_target_lane_id_ugv:
                target_lane_id = last_chosen_lane_id_ugv
            else:
                target_lane_id = current_lane_info.cur_lane_id

            last_chosen_lane_id_ugv = target_lane_id

            rospy.loginfo("final target lane id %d" % target_lane_id)

            if this_target_lane_id_ugv != last_target_lane_id_ugv or current_lane_info.cur_lane_id == last_target_lane_id_ugv:
                lane_change_consideration_start_time = 0
                return 'cancel_intention'

            is_ready, speed_upper_limit_merge = merge_priority_decider(this_target_lane_id_ugv, user_data.obstacles_list, user_data.pose_data, available_lanes)

            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data,
                                                                       certain_speed_limit=speed_upper_limit_merge)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)

            # 避免找到身后的目标路径
            if desired_length > 0:
                reference_path = points_filler(user_data.lane_list, current_lane_info.cur_lane_id, -1, available_lanes,
                                               desired_length)
            else:
                reference_path = []

            obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id, target_lane_id, next_lane_id)

            output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit, reference_path)

            if is_ready and (rospy.get_time() - lane_change_consideration_start_time) > TIME_LANE_CHANGE_CONSIDERATION:
                lane_change_calm_down_start_time = rospy.get_time()
                return "ready_to_change_lane"

            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())

            # if the vehicle on the surrounding lanes is about to cut into this lane. decelerate.


class LaneChanging(smach.State):
    """

    """
    def __init__(self):
        smach.State.__init__(self, outcomes=['lane_change_completed', 'lane_change_cancelled'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in LaneChanging")

            global lane_change_calm_down_start_time, last_chosen_lane_id_ugv
            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            if current_lane_info.cur_lane_id == -1 or current_lane_info.cur_priority <= 0:
                # 的找不到当前车道或者当前车道优先级小，进入merge
                return 'lane_change_cancelled'

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)
            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data, 'lane_follow',
                                                                current_lane_info, available_lanes)

            last_chosen_lane_id_ugv = target_lane_id

            # if target_lane_id != last_target_lane_id_ugv and (rospy.get_time() - lane_change_calm_down_start_time) > TIME_LANE_CHANGE_CALM_DOWN:
            #         return 'lane_change_cancelled'

            if last_target_lane_id_ugv == current_lane_info.cur_lane_id:
                return 'lane_change_completed'


            rospy.loginfo("final target lane id %d" % target_lane_id)
            is_ready, speed_upper_limit_merge = merge_priority_decider(last_target_lane_id_ugv, user_data.obstacles_list,
                                                                       user_data.pose_data, available_lanes)
            if not is_ready:
                'lane_change_cancelled'

            if (rospy.get_time() - lane_change_calm_down_start_time) > TIME_LANE_CHANGE_DEADLINE:
                return 'lane_change_cancelled'

            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       last_target_lane_id_ugv, user_data.pose_data,
                                                                       certain_speed_limit=speed_upper_limit_merge)

            desired_length = desired_length_decider(available_lanes, last_target_lane_id_ugv, speed_upper_limit)

            # 避免找到身后的目标路径
            if desired_length > 0:
                reference_path = points_filler(user_data.lane_list, last_target_lane_id_ugv, next_lane_id, available_lanes,
                                               desired_length)
            else:
                reference_path = []

            obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id, last_target_lane_id_ugv, next_lane_id)

            output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit, reference_path)
            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())

            # if the vehicle on the surrounding lanes is about to cut into this lane. decelerate.


class FindRecoverySolution(smach.State):
    """
    find a solution step by step.
    first, bypass the dynamic obstacles.
    then, rule out priority
    finally, rule out traffic restrictions.
    """

    def __init__(self):
        smach.State.__init__(self, outcomes=['need_to_change_lane', 'back_to_normal'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        rospy.loginfo("currently in FindRecoverySolution")

        user_data_updater(user_data)
        pass


class LaneChangePreparingErrorRecovery(smach.State):
    """
    enter this state when there is a need to change lane.
    select the target lane considering mission and obstacles.
    then, choose a target interval on the target lane.
    return to main state when there is no need to change lane.
    """

    def __init__(self):
        smach.State.__init__(self, outcomes=['cancel_intention', 'ready_to_change_lane'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        rospy.loginfo("currently in LaneChangePreparingErrorRecovery")
        user_data_updater(user_data)
        pass


class LaneChangingErrorRecovery(smach.State):
    """
    When lane change completes, move back to the original lane.
    """

    def __init__(self):
        smach.State.__init__(self, outcomes=['lane_change_completed', 'lane_change_cancelled'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        rospy.loginfo("currently in LaneChangingErrorRecovery")
        user_data_updater(user_data)
        pass


#########################################
# INTERSECTION
#########################################
class ApproachIntersection(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['with_lights', 'without_lights', 'exit_intersection'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):

        while not rospy.is_shutdown():
            rospy.loginfo("currently in ApproachIntersection")

            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            if current_lane_info.cur_lane_id == -1 or current_lane_info.cur_priority <= 0:
                # 的找不到当前车道或者当前车道优先级小，进入merge
                return 'exit_intersection'

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)

            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data, 'lane_follow',
                                                                current_lane_info, available_lanes)


            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data,
                                                                       certain_speed_limit=SPEED_UPPER_LIMIT_APPROACH)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)

            # 避免找到身后的目标路径
            if desired_length > 0:
                reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id, available_lanes,
                                               desired_length)
            else:
                reference_path = []

            obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                          target_lane_id, next_lane_id)

            # if the vehicle on the surrounding lanes is about to cut into this lane. decelerate.
            output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit,
                          reference_path)

            if current_lane_info.dist_to_next_stop < 15 and current_lane_info.next_stop_type == OBSERVE_TRAFFIC_LIGHT:
                return 'with_lights'
            # if current_lane_info.dist_to_next_stop < 15 and current_lane_info.next_stop_type != OBSERVE_TRAFFIC_LIGHT:
            #     return 'without_lights'
            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


class CreepToIntersectionWithLights(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['enter'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in CreepToIntersectionWithLights")

            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)

            if current_lane_info.cur_turn_type:
                return 'enter'

            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data,
                                                                'lane_follow',
                                                                current_lane_info, available_lanes)


            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data,
                                                                       certain_speed_limit=SPEED_UPPER_LIMIT_INTERSECTION)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)

            lanes_of_interest = lanes_of_interest_selector(user_data.lane_list, user_data.pose_data, 'intersection', available_lanes, target_lane_id, next_lane_id)

            is_ready = initial_priority_decider(lanes_of_interest, user_data.obstacles_list)

            if next_lane_id != -1:
                next_turn_type = user_data.lane_list[next_lane_id].turn
                light_status = GREEN
                if next_turn_type in user_data.lights_list.keys():
                    if user_data.lights_list[next_turn_type] is not None:
                        light_status = user_data.lights_list[next_turn_type].color
                if light_status != GREEN:
                    desired_length = min(desired_length, available_lanes[current_lane_info.cur_lane_id].after_length)

            if is_ready:
                # 避免找到身后的目标路径
                if desired_length > 0:
                    reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id, available_lanes,
                                                   desired_length)
                else:
                    reference_path = []

                obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                              target_lane_id, next_lane_id)

                output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit,
                              reference_path, light_detection_switch = 1)
            else:
                speed_upper_limit = 0

                obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                              target_lane_id, next_lane_id)

                output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit,
                              reference_path=[], light_detection_switch = 1)


            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


class CreepToIntersectionWithoutLights(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['enter'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in CreepToIntersectionWithoutLights")

            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)

            if current_lane_info.cur_turn_type:
                return 'enter'

            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data,
                                                                'lane_follow',
                                                                current_lane_info, available_lanes)


            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data,
                                                                       certain_speed_limit=SPEED_UPPER_LIMIT_INTERSECTION)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)

            lanes_of_interest = lanes_of_interest_selector(user_data.lane_list, user_data.pose_data, 'intersection',
                                                           available_lanes, target_lane_id, next_lane_id)

            is_ready = initial_priority_decider(lanes_of_interest, user_data.obstacles_list)

            if is_ready:
                # 避免找到身后的目标路径
                if desired_length > 0:
                    reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id, available_lanes,
                                                   desired_length)
                else:
                    reference_path = []

                obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                              target_lane_id, next_lane_id)

                output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit,
                              reference_path)
            else:
                speed_upper_limit = 0

                obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                              target_lane_id, next_lane_id)

                output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit,
                              reference_path=[])

            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


class EnterIntersection(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['pass'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in EnterIntersection")

            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)

            if not current_lane_info.cur_turn_type:
                return 'pass'

            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data,
                                                                'lane_follow',
                                                                current_lane_info, available_lanes)


            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data,
                                                                       certain_speed_limit=SPEED_UPPER_LIMIT_INTERSECTION)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)

            lanes_of_interest = lanes_of_interest_selector(user_data.lane_list, user_data.pose_data, 'intersection',
                                                           available_lanes, target_lane_id, next_lane_id)

            is_ready = initial_priority_decider(lanes_of_interest, user_data.obstacles_list)

            if is_ready:
                # 避免找到身后的目标路径
                if desired_length > 0:
                    reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id, available_lanes,
                                                   desired_length)
                else:
                    reference_path = []

                obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                              target_lane_id, next_lane_id)

                output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit,
                              reference_path)
            else:
                speed_upper_limit = 0

                obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                              target_lane_id, next_lane_id)

                output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit,
                              reference_path=[])

            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


class PassIntersection(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in PassIntersection")
            return 'succeeded'


#########################################
# STOP LINE
#########################################
class ApproachStopLine(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['near_stop', 'exit_stop'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        user_data_updater(user_data)

        current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)
        global last_stop_x
        last_stop_x = current_lane_info.next_stop_x

        while not rospy.is_shutdown():
            rospy.loginfo("currently in ApproachStopLine")

            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            if current_lane_info.cur_lane_id == -1 or current_lane_info.cur_priority <= 0:
                # 的找不到当前车道或者当前车道优先级小，进入merge
                return 'exit_stop'

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)

            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data, 'lane_follow',
                                                                current_lane_info, available_lanes)


            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data,
                                                                       certain_speed_limit=SPEED_UPPER_LIMIT_APPROACH)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)
            desired_length = min(desired_length, available_lanes[current_lane_info.cur_lane_id].after_length)

            # 避免找到身后的目标路径
            if desired_length > 0:
                reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id, available_lanes,
                                               desired_length)
            else:
                reference_path = []

            obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                          target_lane_id, next_lane_id)

            # if the vehicle on the surrounding lanes is about to cut into this lane. decelerate.
            output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit,
                          reference_path)

            if current_lane_info.dist_to_next_stop < 15 and current_lane_info.next_stop_type == STOP_GIVE_WAY:
                return 'near_stop'

            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


class CreepToStopLine(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['stopped'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in CreepToStopLine")

            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)

            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data,
                                                                'lane_follow',
                                                                current_lane_info, available_lanes)


            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data,
                                                                       certain_speed_limit=SPEED_UPPER_LIMIT_INTERSECTION)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)
            desired_length = min(desired_length, available_lanes[current_lane_info.cur_lane_id].after_length)

            lanes_of_interest = lanes_of_interest_selector(user_data.lane_list, user_data.pose_data, 'intersection',
                                                           available_lanes, target_lane_id, next_lane_id)

            is_ready = initial_priority_decider(lanes_of_interest, user_data.obstacles_list)

            if is_ready:
                # 避免找到身后的目标路径
                if desired_length > 0:
                    reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id, available_lanes,
                                                   desired_length)
                else:
                    reference_path = []

                obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                              target_lane_id, next_lane_id)

                output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit,
                              reference_path)
            else:
                speed_upper_limit = 0

                obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                              target_lane_id, next_lane_id)

                output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit,
                              reference_path=[])

            if current_lane_info.dist_to_next_stop < STOP_BASE_LENGTH and user_data.pose_data.mVf < 0.5:
                return 'stopped'

            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


class PassStopLine(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['pass'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in PassStopLine")

            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)

            global wait_duration, action_done, last_stop_x
            wait_duration += 1
            if abs(current_lane_info.next_stop_x - last_stop_x) > EPS:
                last_stop_x = -1
                action_done = False
                wait_duration = 0
                return 'pass'

            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data,
                                                                'lane_follow',
                                                                current_lane_info, available_lanes)


            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data,
                                                                       certain_speed_limit=SPEED_UPPER_LIMIT_INTERSECTION)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)
            if wait_duration < 20 and action_done == False:
                desired_length = min(desired_length, available_lanes[current_lane_info.cur_lane_id].after_length)
            else:
                wait_duration = 0
                action_done = True

            lanes_of_interest = lanes_of_interest_selector(user_data.lane_list, user_data.pose_data, 'intersection',
                                                           available_lanes, target_lane_id, next_lane_id)

            is_ready = initial_priority_decider(lanes_of_interest, user_data.obstacles_list)

            if is_ready:
                # 避免找到身后的目标路径
                if desired_length > 0:
                    reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id, available_lanes,
                                                   desired_length)
                else:
                    reference_path = []

                obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                              target_lane_id, next_lane_id)

                output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit,
                              reference_path)
            else:
                speed_upper_limit = 0

                obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                              target_lane_id, next_lane_id)

                output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit,
                              reference_path=[])

            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


#########################################
# MERGE AND ACROSS
#########################################
class CreepForOpportunity(smach.State):
    """
    consider the target lane which will merge into and the lanes which will across.
    """

    def __init__(self):
        smach.State.__init__(self, outcomes=['ready', 'back_to_normal', 'no_mission'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in CreepForOpportunity")

            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            # 如果找到当前车道，且当前车道优先级为正
            if current_lane_info.cur_priority > 0:
                return 'back_to_normal'

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)
            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data, 'merge',
                                                                current_lane_info, available_lanes)


            if target_lane_id == -1:
                # merge 都找不到目标车道，则重新全局规划
                re_global_planning_caller()
                return 'no_mission'

            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data, merge_decelerate=1)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit, scenario='merge')

            lanes_of_interest = lanes_of_interest_selector(user_data.lane_list, user_data.pose_data, 'merge',
                                                           available_lanes, target_lane_id, next_lane_id)
            is_ready = initial_priority_decider(lanes_of_interest, user_data.obstacles_list)

            if is_ready:
                return 'ready'
            else:
                speed_upper_limit = 0
                # 避免找到身后的目标路径
            if desired_length > 0:
                reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id,
                                               available_lanes,
                                               desired_length)
            else:
                reference_path = []

            obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                          target_lane_id, next_lane_id)

            output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit, reference_path)

            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


class ExecuteMerge(smach.State):
    """

    """
    def __init__(self):
        smach.State.__init__(self, outcomes=['back_to_normal', 'break'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):

        while not rospy.is_shutdown():
            rospy.loginfo("currently in ExecuteMerge")

            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            # 如果找到当前车道，且当前车道优先级为正
            if current_lane_info.cur_priority > 0:
                return 'back_to_normal'

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)
            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data, 'merge',
                                                                current_lane_info, available_lanes)


            if target_lane_id == -1:
                return 'break'

            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data,
                                                                       merge_decelerate=5)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit, scenario='merge')

            lanes_of_interest = lanes_of_interest_selector(user_data.lane_list, user_data.pose_data, 'merge',
                                                           available_lanes, target_lane_id, next_lane_id)
            is_ready = initial_priority_decider(lanes_of_interest, user_data.obstacles_list)
            if is_ready:
                # 避免找到身后的目标路径
                if desired_length > 0:
                    reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id, available_lanes,
                                                   desired_length)
                else:
                    reference_path = []

                obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                              target_lane_id, next_lane_id)

                output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit, reference_path)
            else:
                speed_upper_limit = 0

                obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                              target_lane_id, next_lane_id)

                output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit, reference_path=[])
                return 'break'
            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


#########################################
# PARKING
#########################################
class DriveAlongLane(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['enter_parking_zone', 'lane_end', 'exit_park'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in DriveAlongLane")

            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            if current_lane_info.cur_lane_id == -1 or current_lane_info.cur_priority <= 0:
                # 的找不到当前车道或者当前车道优先级小，进入merge
                return 'exit_park'

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)

            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data, 'lane_follow',
                                                                current_lane_info, available_lanes)

            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)
            if desired_length > 0:
                reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id,
                                               available_lanes,
                                               desired_length)
            else:
                reference_path = []

            obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                          target_lane_id, next_lane_id)

            output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit, reference_path)

            if parking_slots_list != {}:
                return 'enter_parking_zone'
            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


class SelectParkingSlot(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['have_empty_slot', 'no_empty_slot'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        rospy.sleep(DECISION_PERIOD)
        user_data_updater(user_data)
        current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)
        if current_lane_info.cur_lane_id != -1:
            history_lanes_recorder(current_lane_info.cur_lane_id)

        temp_lane = user_data.lane_list[current_lane_info.cur_lane_id]
        points_x, points_y = [], []
        parking_slot_sorted = []
        for j in range(len(temp_lane.points)):
            points_x.append(temp_lane.points[j].x)
            points_y.append(temp_lane.points[j].y)
        points_num = len(points_x)
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        for key in user_data.parking_slots_list.keys():
            print(key,user_data.parking_slots_list[key].lane_id,current_lane_info.cur_lane_id)
            if user_data.parking_slots_list[key].lane_id == current_lane_info.cur_lane_id:
                print("aaaaaaaaa")
                res = lane_projection(points_x, points_y, points_num, user_data.parking_slots_list[key].lane_projection_point.x,
                                                 user_data.parking_slots_list[key].lane_projection_point.y)
                print(res[-2], current_lane_info.before_length)
                if current_lane_info.before_length + 10 > res[-2] >= current_lane_info.before_length:
                    parking_slot_sorted.append([res[-2],key])
        parking_slot_sorted.sort()
        print(parking_slot_sorted)
        while not rospy.is_shutdown():
            rospy.loginfo("currently in SelectParkingSlot")

            global target_parking_slot, target_parking_slot_projection
            target_parking_slot = []
            target_parking_slot_projection = []
            target_slot_id = 0
            rospy.loginfo("available parking slots ids %s" % list(user_data.parking_slots_list.keys()))
            rospy.loginfo("mission thing id %d" % mission_ahead.missionThingId)
            if mission_ahead.missionThingId in user_data.parking_slots_list.keys():
                rospy.loginfo("park into a certain parking slot")
                target_slot_id = mission_ahead.missionThingId
                target_parking_slot = user_data.parking_slots_list[mission_ahead.missionThingId].points
                target_parking_slot_projection.append(user_data.parking_slots_list[target_slot_id].lane_projection_point.x)
                target_parking_slot_projection.append(user_data.parking_slots_list[target_slot_id].lane_projection_point.y)
                return 'have_empty_slot'
            else:
                # missionThing为“parkingArea" 选择车位
                rospy.loginfo("available parking slots ids %s" % list(user_data.parking_slots_list.keys()))
                rospy.loginfo("need to choose a parking slot")
                extend_distance = 4
                parking_area = []
                for i in range(len(target_parking_area)):
                    parking_area.append([target_parking_area[i].x,target_parking_area[i].y])
                parking_area = Polygon(parking_area)
                obstacle_in_area = []
                for key in user_data.obstacles_list.keys():
                    for point in user_data.obstacles_list[key].cur_bounding_points:
                        new_point = Point([point.x,point.y])
                        if parking_area.contains(new_point):
                            new_obstacle = []
                            for obstacle_points in user_data.obstacles_list[key].cur_bounding_points:
                                new_obstacle.append([obstacle_points.x, obstacle_points.y])
                            obstacle_in_area.append(Polygon(new_obstacle))
                            break
                print(parking_slot_sorted)
                print(obstacle_in_area)
                for item in parking_slot_sorted:
                    key = item[1]
                    slot = user_data.parking_slots_list[key]
                    slot_list = [[slot.points[0].x, slot.points[0].y], [slot.points[1].x, slot.points[1].y], [slot.points[2].x, slot.points[2].y], [slot.points[3].x, slot.points[3].y]]
                    lot_shape = Polygon(slot_list)
                    if parking_area.contains(lot_shape):
                        if user_data.parking_slots_list[key].slot_type == PARALLEL_SLOT:
                            heading = [slot_list[3][0] - slot_list[0][0], slot_list[3][1] - slot_list[0][1]]
                            heading = heading / np.linalg.norm(heading)
                            new_bound1 = extend_distance * heading + slot_list[2]
                            new_bound2 = extend_distance * heading + slot_list[3]
                            lotShape1 = Polygon([slot_list[0], slot_list[1], new_bound1, new_bound2])
                            occupied = 0
                            for obs in obstacle_in_area:
                                if lotShape1.contains(obs) or lotShape1.intersects(obs) or obs.contains(lotShape1):
                                    occupied = 1
                            if occupied == 0:
                                target_slot_id = key
                                break
                        else:
                            occupied = 0
                            for obs in obstacle_in_area:
                                if lot_shape.contains(obs) or lot_shape.intersects(obs) or obs.contains(lot_shape):
                                    occupied = 1
                            if occupied == 0:
                                target_slot_id = key
                                break
                if not target_slot_id:
                    rospy.loginfo('no parking slot available')
                    return 'no_empty_slot'
                else:
                    target_parking_slot = user_data.parking_slots_list[target_slot_id].points
                    target_parking_slot_projection.append(user_data.parking_slots_list[target_slot_id].lane_projection_point.x)
                    target_parking_slot_projection.append(user_data.parking_slots_list[target_slot_id].lane_projection_point.y)
                    rospy.loginfo("selected parking slot id %d points %s " % (key, target_parking_slot))
                    return 'have_empty_slot'


class DriveAndStopInFront(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['finished', 'exit_park'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        user_data_updater(user_data)

        current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)
        rospy.loginfo("current lane id %f" % current_lane_info.cur_lane_id)
        if current_lane_info.cur_lane_id != -1:
            history_lanes_recorder(current_lane_info.cur_lane_id)

        global parking_lane_id
        parking_lane_id = current_lane_info.cur_lane_id

        temp_lane = user_data.lane_list[parking_lane_id]
        points_x, points_y = [], []
        for j in range(len(temp_lane.points)):
            points_x.append(temp_lane.points[j].x)
            points_y.append(temp_lane.points[j].y)
        points_num = len(points_x)
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        project_result = lane_projection(points_x, points_y, points_num, target_parking_slot_projection[0],
                                         target_parking_slot_projection[1])
        rospy.loginfo("parking slot projection x %f" % project_result[0])
        rospy.loginfo("parking slot projection y %f" % project_result[1])
        rospy.loginfo("target parking slot center %f" % target_parking_slot_projection[0])
        rospy.loginfo("target parking slot center %f" % target_parking_slot_projection[1])

        while not rospy.is_shutdown():
            rospy.loginfo("currently in DriveAndStopInFront")
            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, user_data.pose_data)

            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            if current_lane_info.cur_lane_id == -1 or current_lane_info.cur_priority <= 0:
                # 的找不到当前车道或者当前车道优先级小，进入merge
                return 'exit_park'

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, user_data.pose_data,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)

            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, user_data.pose_data, 'lane_follow',
                                                                current_lane_info, available_lanes)

            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)

            if project_result[5] - available_lanes[target_lane_id].before_length < desired_length:
                desired_length = project_result[5] - available_lanes[target_lane_id].before_length

            rospy.loginfo("drivable length %f" % available_lanes[target_lane_id].front_drivable_length)
            rospy.loginfo("desired length %f" % desired_length)

            if desired_length > 0:
                reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id,
                                               available_lanes,
                                               desired_length)
            else:
                reference_path = []
            dis_to_slot = math.sqrt(math.pow(user_data.pose_data.mapX - project_result[0], 2) + math.pow(user_data.pose_data.mapY - project_result[1], 2))
            print(dis_to_slot)
            if dis_to_slot < STOP_BASE_LENGTH and user_data.pose_data.mVf < 0.5:
                return 'finished'

            obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                          target_lane_id, next_lane_id)

            output_filler(REF_PATH_FOLLOW, user_data.obstacles_list, speed_upper_limit, speed_lower_limit, reference_path)

            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


class ExecutePark(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        # 计算目标车位中心点到泊车目标车位的投影点
        user_data_updater(user_data)

        virtual_pose = Pose()
        virtual_pose.mapX = user_data.pose_data.mapX
        virtual_pose.mapY = user_data.pose_data.mapY
        virtual_pose.mapHeading = user_data.pose_data.mapHeading
        rospy.loginfo("virtual pose x %f" % virtual_pose.mapX)
        rospy.loginfo("virtual pose y %f" % virtual_pose.mapY)
        rospy.loginfo("virtual pose heading %f" % virtual_pose.mapHeading)

        while not rospy.is_shutdown():
            rospy.loginfo("currently in ExecutePark")
            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, virtual_pose)
            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, virtual_pose,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)

            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, virtual_pose, 'lane_follow',
                                                                current_lane_info, available_lanes)

            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)

            if desired_length > 0:
                reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id,
                                               available_lanes,
                                               desired_length)
            else:
                reference_path = []

            global planning_feedback
            if planning_feedback == PARKING_DONE:
                planning_feedback = 0
                return 'succeeded'

            obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                          target_lane_id, next_lane_id)

            output_filler(PARK, user_data.obstacles_list, speed_upper_limit, speed_lower_limit, reference_path,
                          selected_parking_lot=target_parking_slot)

            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


class PoseCheck(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['okay', 'need_to_adjust'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in PoseCheck")
            if 1:
                return 'okay'


class RePark(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in RePark")
            pass


class AwaitMission(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['continue'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):

        mission_finished_caller()
        output_filler(scenario=NONE)
        rospy.sleep(10)
        while not rospy.is_shutdown():
            rospy.loginfo("currently in AwaitMission")
            if mission_ahead.missionType is None:
                output_filler(scenario=NONE)
                rospy.sleep(DECISION_PERIOD)
                continue
            else:
                return 'continue'


class ExitParkingSlot(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['back_to_lane'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        # 计算目标车位中心点到泊车目标车位的投影点
        user_data_updater(user_data)
        temp_lane = user_data.lane_list[parking_lane_id]
        points_x, points_y = [], []
        for j in range(len(temp_lane.points)):
            points_x.append(temp_lane.points[j].x)
            points_y.append(temp_lane.points[j].y)
        points_num = len(points_x)
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        project_result = lane_projection(points_x, points_y, points_num, target_parking_slot_projection[0],
                                         target_parking_slot_projection[1])

        virtual_pose = Pose()
        virtual_pose.mapX = project_result[0]
        virtual_pose.mapY = project_result[1]
        virtual_pose.mapHeading = math.atan2((temp_lane.points[project_result[2] + 1].y -
                                              temp_lane.points[project_result[2]].y),
                                             (temp_lane.points[project_result[2] + 1].x -
                                              temp_lane.points[project_result[2]].x))
        rospy.loginfo("virtual pose x %f" % virtual_pose.mapX)
        rospy.loginfo("virtual pose y %f" % virtual_pose.mapY)
        rospy.loginfo("virtual pose heading %f" % virtual_pose.mapHeading)

        while not rospy.is_shutdown():
            rospy.loginfo("currently in ExitParkingSlot")

            start_time = rospy.get_time()
            rospy.loginfo("start time %f" % start_time)
            user_data_updater(user_data)

            current_lane_info, available_lanes = current_lane_selector(user_data.lane_list, virtual_pose)
            if current_lane_info.cur_lane_id != -1:
                history_lanes_recorder(current_lane_info.cur_lane_id)

            available_lanes, current_lane_info = available_lanes_selector(user_data.lane_list, virtual_pose,
                                                                          user_data.obstacles_list, current_lane_info,
                                                                          available_lanes)

            # compare the reward value among the surrounding lanes.
            target_lane_id, next_lane_id = target_lane_selector(user_data.lane_list, virtual_pose, 'lane_follow',
                                                                current_lane_info, available_lanes)

            speed_upper_limit, speed_lower_limit = speed_limit_decider(user_data.lane_list, available_lanes,
                                                                       current_lane_info,
                                                                       target_lane_id, user_data.pose_data)

            desired_length = desired_length_decider(available_lanes, target_lane_id, speed_upper_limit)

            if desired_length > 0:
                reference_path = points_filler(user_data.lane_list, target_lane_id, next_lane_id,
                                               available_lanes,
                                               desired_length)
            else:
                reference_path = []

            global planning_feedback
            if planning_feedback == EXIT_PARKING_DONE:
                planning_feedback = 0
                return 'back_to_lane'

            obstacle_of_interest_selector(user_data.obstacles_list, available_lanes, current_lane_info.cur_lane_id,
                                          target_lane_id, next_lane_id)

            output_filler(EXIT_PARK, user_data.obstacles_list, speed_upper_limit, speed_lower_limit, reference_path,
                          selected_parking_lot=target_parking_slot)

            end_time = rospy.get_time()
            rospy.sleep(DECISION_PERIOD + start_time - end_time)
            rospy.loginfo("end time %f" % rospy.get_time())


class MarkParkingSpot(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        rospy.loginfo("currently in MarkParkingSpot")
        pass


class ReturnToLane(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        rospy.loginfo("currently in ReturnToLane")
        pass


class ReGlobalPlan(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['continue', 'need_to_turn_around'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        rospy.loginfo("currently in ReGlobalPlan")
        pass


class TurnAround(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        rospy.loginfo("currently in TurnAround")
        pass


class Reverse(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['okay_to_turn_around'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        rospy.loginfo("currently in Reverse")
        pass


#########################################
# EMERGENCY BRAKE
#########################################
class ConditionJudge(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['brakeOn'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):

        while not rospy.is_shutdown():
            # rospy.loginfo("currently in ConditionJudge")
            # if lane_list == {}:
            #     rospy.loginfo("emergency brake because of no lane")
            #     return 'brakeOn'
            rospy.sleep(DECISION_PERIOD)
            global scenario_error_handle
            user_data_updater(user_data)
            if user_data.pose_data is None:
                continue
            if user_data.pose_data.localizationMode < 0:
                scenario_error_handle = NONE
                rospy.loginfo("StopImmediately because of no global pose")
            else:
                scenario_error_handle = REF_PATH_FOLLOW



class StopImmediately(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['brakeOff'],
                             input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                         'lights_list', 'pose_data', 'parking_slots_list'],
                             output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                          'lights_list', 'pose_data', 'parking_slots_list']
                             )

    def execute(self, user_data):
        while not rospy.is_shutdown():
            rospy.loginfo("currently in StopImmediately")
            rospy.sleep(DECISION_PERIOD)


#########################################
# RE-GLOBAL PLANNING
#########################################
# class EmergencyBrake(smach.State):
#     def __init__(self):
#         smach.State.__init__(self, outcomes=['brakeOn'],
#                              input_keys=['lane_list', 'obstacles_list', 'signs_data',
#                                          'lights_list', 'pose_data', 'parking_slots_list'],
#                              output_keys=['lane_list', 'obstacles_list', 'signs_data',
#                                           'lights_list', 'pose_data', 'parking_slots_list']
#                              )
#
#     def execute(self, user_data):
#         while not rospy.is_shutdown():
#             rospy.loginfo("currently in EmergencyBrake")
#             if lane_list == {}:
#                 rospy.loginfo("emergency brake because of no lane")
#             rospy.sleep(DECISION_PERIOD)
#
#
# class StopImmediately(smach.State):
#     def __init__(self):
#         smach.State.__init__(self, outcomes=['brakeOff'],
#                              input_keys=['lane_list', 'obstacles_list', 'signs_data',
#                                          'lights_list', 'pose_data', 'parking_slots_list'],
#                              output_keys=['lane_list', 'obstacles_list', 'signs_data',
#                                           'lights_list', 'pose_data', 'parking_slots_list']
#                              )
#
#     def execute(self, user_data):
#         while not rospy.is_shutdown():
#             rospy.loginfo("currently in Await")



#########################################
# GEAR SWITCH
#########################################
# class GearSwitch(smach.State):
#     def __init__(self):
#         smach.State.__init__(self, outcomes=['succeeded'])
#
#     def execute(self, user_data):
#         while not rospy.is_shutdown():
#             global ready_to_go
#             rospy.loginfo("cur  gear11111111111:%d" % current_gear)
#             rospy.loginfo("ref  gear11111111111:%d" % reference_gear)
#             if current_gear != reference_gear:
#                 ready_to_go = False
#             else:
#                 ready_to_go = True
#
#             rospy.sleep(DECISION_PERIOD)
#             if mission_completed:
#                 return 'succeeded'


def main():
    rospy.init_node('decision_smach')

    listener()
    server()
    # Create the top level SMACH state machine
    sm_top = smach.StateMachine(outcomes=[])
    sm_top.userdata.lane_list = {}
    sm_top.userdata.obstacles_list = {}
    sm_top.userdata.signs_data = None
    sm_top.userdata.lights_list = None
    sm_top.userdata.pose_data = None
    sm_top.userdata.parking_slots_list = {}

    # Open the container
    with sm_top:
        # Create the sub SMACH state machine
        sm_con = smach.Concurrence(outcomes=['outcome5'], default_outcome='outcome5',
                                   input_keys=['lane_list', 'obstacles_list', 'signs_data',
                                               'lights_list', 'pose_data', 'parking_slots_list'],
                                   output_keys=['lane_list', 'obstacles_list', 'signs_data',
                                                'lights_list', 'pose_data', 'parking_slots_list']
                                   )
        # {'outcome5':{'FOO': 'outcome2','BAR': 'outcome1'}}表示 FOO 和 BAR 输出都要满足条件才会输出 outcome5

        # Open the container
        with sm_con:
            sm_con_scenario = smach.StateMachine(outcomes=['outcome4'],
                                                 input_keys=['lane_list', 'obstacles_list',
                                                             'signs_data', 'lights_list', 'pose_data', 'parking_slots_list'],
                                                 output_keys=['lane_list', 'obstacles_list',
                                                              'signs_data',
                                                              'lights_list', 'pose_data', 'parking_slots_list']
                                                 )

            with sm_con_scenario:
                sm_scenario_startup = smach.StateMachine(outcomes=['in_lane_driving', 'merge_and_across', 'park'],
                                                         input_keys=['lane_list',
                                                                     'obstacles_list', 'signs_data', 'lights_list',
                                                                     'pose_data', 'parking_slots_list'],
                                                         output_keys=['lane_list',
                                                                      'obstacles_list', 'signs_data',
                                                                      'lights_list', 'pose_data', 'parking_slots_list']
                                                         )
                with sm_scenario_startup:
                    smach.StateMachine.add('STARTUP_CHECK', StartupCheck(), transitions={'ready': 'EXECUTE_STARTUP'})
                    smach.StateMachine.add('EXECUTE_STARTUP', Startup(),
                                           transitions={'in_lane_driving': 'in_lane_driving',
                                                        'merge_and_across': 'merge_and_across',
                                                        'park': 'park'})
                smach.StateMachine.add('STARTUP', sm_scenario_startup, transitions={'in_lane_driving': 'LANE_FOLLOW',
                                                                                    'merge_and_across': 'MERGE_AND_ACROSS',
                                                                                    'park': 'PARK'})

                sm_scenario_lane_follow = smach.StateMachine(outcomes=['park', 'intersection', 'merge_and_across', 'finished', 'stop_line'],
                                                             input_keys=['lane_list',
                                                                         'obstacles_list', 'signs_data', 'lights_list',
                                                                         'pose_data', 'parking_slots_list'],
                                                             output_keys=['lane_list',
                                                                          'obstacles_list', 'signs_data',
                                                                          'lights_list', 'pose_data', 'parking_slots_list']
                                                             )
                with sm_scenario_lane_follow:
                    smach.StateMachine.add('IN_LANE_DRIVING', InLaneDriving(), transitions={'park': 'park',
                                                                                            'intersection': 'intersection',
                                                                                            'merge_and_across': 'merge_and_across',
                                                                                            'need_to_change_lane': 'LANE_CHANGE_PREPARING',
                                                                                            'error': 'ERROR_RECOVERY',
                                                                                            'finished': 'finished',
                                                                                            'stop_line': 'stop_line'})
                    smach.StateMachine.add('LANE_CHANGE_PREPARING', LaneChangePreparing(),
                                           transitions={'cancel_intention': 'IN_LANE_DRIVING',
                                                        'ready_to_change_lane': 'LANE_CHANGING'})
                    smach.StateMachine.add('LANE_CHANGING', LaneChanging(),
                                           transitions={'lane_change_completed': 'IN_LANE_DRIVING',
                                                        'lane_change_cancelled': 'IN_LANE_DRIVING'})
                    sm_scenario_lane_follow_error_recovery = smach.StateMachine(outcomes=['back_to_normal'],
                                                                                input_keys=['lane_list',
                                                                                            'obstacles_list',
                                                                                            'signs_data', 'lights_list',
                                                                                            'pose_data', 'parking_slots_list'],
                                                                                output_keys=['lane_list',
                                                                                             'obstacles_list',
                                                                                             'signs_data',
                                                                                             'lights_list', 'pose_data', 'parking_slots_list']
                                                                                )
                    with sm_scenario_lane_follow_error_recovery:
                        smach.StateMachine.add('FIND_RECOVERY_SOLUTION', FindRecoverySolution(),
                                               transitions={'back_to_normal': 'back_to_normal',
                                                            'need_to_change_lane': 'LANE_CHANGE_PREPARING_RECOVERY'})
                        smach.StateMachine.add('LANE_CHANGE_PREPARING_RECOVERY', LaneChangePreparingErrorRecovery(),
                                               transitions={'cancel_intention': 'FIND_RECOVERY_SOLUTION',
                                                            'ready_to_change_lane': 'LANE_CHANGE_RECOVERY'})
                        smach.StateMachine.add('LANE_CHANGE_RECOVERY', LaneChangingErrorRecovery(),
                                               transitions={'lane_change_completed': 'LANE_CHANGE_PREPARING_RECOVERY',
                                                            'lane_change_cancelled': 'LANE_CHANGE_PREPARING_RECOVERY'})
                    smach.StateMachine.add('ERROR_RECOVERY', sm_scenario_lane_follow_error_recovery,
                                           transitions={'back_to_normal': 'IN_LANE_DRIVING'})
                smach.StateMachine.add('LANE_FOLLOW', sm_scenario_lane_follow, transitions={'park': 'PARK',
                                                                                            'intersection': 'INTERSECTION',
                                                                                            'merge_and_across': 'MERGE_AND_ACROSS',
                                                                                            'finished': 'STARTUP',
                                                                                            'stop_line': 'STOP_GIVE_WAY'})

                sm_scenario_intersection = smach.StateMachine(outcomes=['succeeded'],
                                                              input_keys=['lane_list',
                                                                          'obstacles_list', 'signs_data', 'lights_list',
                                                                          'pose_data', 'parking_slots_list'],
                                                              output_keys=['lane_list',
                                                                           'obstacles_list', 'signs_data',
                                                                           'lights_list', 'pose_data', 'parking_slots_list']
                                                              )
                with sm_scenario_intersection:
                    smach.StateMachine.add('APPROACH_INTERSECTION', ApproachIntersection(),
                                           transitions={'with_lights': 'CREEP_TO_INTERSECTION_WITH_LIGHTS',
                                                        'without_lights': 'CREEP_TO_INTERSECTION_WITHOUT_LIGHTS',
                                                        'exit_intersection': 'succeeded'})
                    smach.StateMachine.add('CREEP_TO_INTERSECTION_WITH_LIGHTS', CreepToIntersectionWithLights(),
                                           transitions={'enter': 'ENTER_INTERSECTION'})
                    smach.StateMachine.add('CREEP_TO_INTERSECTION_WITHOUT_LIGHTS', CreepToIntersectionWithoutLights(),
                                           transitions={'enter': 'ENTER_INTERSECTION'})
                    smach.StateMachine.add('ENTER_INTERSECTION', EnterIntersection(),
                                           transitions={'pass': 'PASS_INTERSECTION'})
                    smach.StateMachine.add('PASS_INTERSECTION', PassIntersection(),
                                           transitions={'succeeded': 'succeeded'})
                smach.StateMachine.add('INTERSECTION', sm_scenario_intersection,
                                       transitions={'succeeded': 'LANE_FOLLOW'})

                sm_scenario_stop = smach.StateMachine(outcomes=['succeeded'],
                                                              input_keys=['lane_list',
                                                                          'obstacles_list', 'signs_data', 'lights_list',
                                                                          'pose_data', 'parking_slots_list'],
                                                              output_keys=['lane_list',
                                                                           'obstacles_list', 'signs_data',
                                                                           'lights_list', 'pose_data',
                                                                           'parking_slots_list']
                                                              )
                with sm_scenario_stop:
                    smach.StateMachine.add('APPROACH_STOP_LINE', ApproachStopLine(),
                                           transitions={'near_stop': 'CREEP_TO_STOP_LINE',
                                                        'exit_stop': 'succeeded'})
                    smach.StateMachine.add('CREEP_TO_STOP_LINE', CreepToStopLine(),
                                           transitions={'stopped': 'PASS_STOP_LINE'})
                    smach.StateMachine.add('PASS_STOP_LINE', PassStopLine(),
                                           transitions={'pass': 'succeeded'})
                smach.StateMachine.add('STOP_GIVE_WAY', sm_scenario_stop,
                                       transitions={'succeeded': 'LANE_FOLLOW'})

                sm_scenario_merge = smach.StateMachine(outcomes=['succeeded', 'restart'],
                                                       input_keys=['lane_list', 'obstacles_list',
                                                                   'signs_data', 'lights_list', 'pose_data', 'parking_slots_list'],
                                                       output_keys=['lane_list',
                                                                    'obstacles_list', 'signs_data',
                                                                    'lights_list', 'pose_data', 'parking_slots_list']
                                                       )
                with sm_scenario_merge:
                    smach.StateMachine.add('CREEP_FOR_OPPORTUNITY', CreepForOpportunity(),
                                           transitions={'ready': 'EXECUTE_MERGE',
                                                        'back_to_normal': 'succeeded',
                                                        'no_mission': 'restart'})
                    smach.StateMachine.add('EXECUTE_MERGE', ExecuteMerge(),
                                           transitions={'back_to_normal': 'succeeded',
                                                        'break': 'CREEP_FOR_OPPORTUNITY'})
                smach.StateMachine.add('MERGE_AND_ACROSS', sm_scenario_merge, transitions={'succeeded': 'LANE_FOLLOW',
                                                                                           'restart': 'STARTUP'})

                sm_scenario_park = smach.StateMachine(outcomes=['mission_continue'],
                                                      input_keys=['lane_list', 'obstacles_list',
                                                                  'signs_data', 'lights_list', 'pose_data', 'parking_slots_list'],
                                                      output_keys=['lane_list', 'obstacles_list',
                                                                   'signs_data',
                                                                   'lights_list', 'pose_data', 'parking_slots_list']
                                                      )
                with sm_scenario_park:
                    smach.StateMachine.add('DRIVE_ALONG_LANE', DriveAlongLane(),
                                           transitions={'enter_parking_zone': 'SELECT_PARKING_SPOT',
                                                        'lane_end': 'RE_GLOBAL_PLAN',
                                                        'exit_park': 'mission_continue'})
                    smach.StateMachine.add('SELECT_PARKING_SPOT', SelectParkingSlot(),
                                           transitions={'have_empty_slot': 'DRIVE_AND_STOP_IN_FRONT',
                                                        'no_empty_slot': 'DRIVE_ALONG_LANE'})
                    smach.StateMachine.add('DRIVE_AND_STOP_IN_FRONT', DriveAndStopInFront(),
                                           transitions={'finished': 'EXECUTE_PARK',
                                                        'exit_park': 'mission_continue'})
                    smach.StateMachine.add('EXECUTE_PARK', ExecutePark(),
                                           transitions={'succeeded': 'POSE_CHECK',
                                                        'failed': 'MARK_PARKING_SPOT'})
                    smach.StateMachine.add('POSE_CHECK', PoseCheck(),
                                           transitions={'okay': 'AWAIT_MISSION',
                                                        'need_to_adjust': 'RE_PARK'})
                    smach.StateMachine.add('RE_PARK', RePark(),
                                           transitions={'succeeded': 'POSE_CHECK'})
                    smach.StateMachine.add('AWAIT_MISSION', AwaitMission(),
                                           transitions={'continue': 'EXIT_PARKING_SPOT'})
                    smach.StateMachine.add('EXIT_PARKING_SPOT', ExitParkingSlot(),
                                           transitions={'back_to_lane': 'mission_continue'})
                    smach.StateMachine.add('MARK_PARKING_SPOT', MarkParkingSpot(),
                                           transitions={'succeeded': 'RETURN_TO_LANE'})
                    smach.StateMachine.add('RETURN_TO_LANE', ReturnToLane(),
                                           transitions={'succeeded': 'DRIVE_ALONG_LANE'})
                    smach.StateMachine.add('RE_GLOBAL_PLAN', ReGlobalPlan(),
                                           transitions={'continue': 'DRIVE_ALONG_LANE',
                                                        'need_to_turn_around': 'TURN_AROUND'})
                    smach.StateMachine.add('TURN_AROUND', TurnAround(),
                                           transitions={'succeeded': 'DRIVE_ALONG_LANE',
                                                        'failed': 'REVERSE'})
                    smach.StateMachine.add('REVERSE', Reverse(),
                                           transitions={'okay_to_turn_around': 'TURN_AROUND'})
                smach.StateMachine.add('PARK', sm_scenario_park, transitions={'mission_continue': 'STARTUP'})

            smach.Concurrence.add('SCENARIO_MANAGER', sm_con_scenario)

            # sm_gear_switch = smach.StateMachine(outcomes=['succeeded'])
            # with sm_gear_switch:
            #     smach.StateMachine.add('GEAR_SWITCH', GearSwitch(), transitions={'succeeded': 'succeeded'})
            # smach.Concurrence.add('GEAR_SWITCH', sm_gear_switch)



            # sm_re_global_Planning = smach.StateMachine(outcomes=['succeeded'],
            #                                            input_keys=['lane_list', 'obstacles_list',
            #                                                        'signs_data', 'lights_list', 'pose_data'],
            #                                            output_keys=['lane_list',
            #                                                         'obstacles_list', 'signs_data',
            #                                                         'lights_list', 'pose_data']
            #                                            )
            # with sm_re_global_Planning:
            #     smach.StateMachine.add('MOVING_FORWARD', ConditionJudge(), transitions={'satisfied': 'STOP'})
            #     smach.StateMachine.add('STOP', StopImmediately(), transitions={'succeeded': 'MOVING_FORWARD'})
            # smach.Concurrence.add('RE_GLOBAL_PLANNING', sm_re_global_Planning)

            sm_emergency_brake = smach.StateMachine(outcomes=['succeeded'],
                                                    input_keys=['lane_list', 'obstacles_list',
                                                                'signs_data', 'lights_list', 'pose_data',
                                                                'parking_slots_list'],
                                                    output_keys=['lane_list', 'obstacles_list',
                                                                 'signs_data',
                                                                 'lights_list', 'pose_data', 'parking_slots_list']
                                                    )
            with sm_emergency_brake:
                smach.StateMachine.add('MOVING_FORWARD', ConditionJudge(), transitions={'brakeOn': 'AWAIT'})
                smach.StateMachine.add('AWAIT', StopImmediately(), transitions={'brakeOff': 'MOVING_FORWARD'})
            smach.Concurrence.add('EMERGENCY_BRAKE', sm_emergency_brake)

        smach.StateMachine.add('FINITE_STATE_MACHINE', sm_con, transitions={'outcome5': 'FINITE_STATE_MACHINE'})

    #

    # Create and start the introspection server
    sis = smach_ros.IntrospectionServer('my_smach_introspection_server', sm_top, '/SM_ROOT')
    sis.start()

    # Execute SMACH plan
    outcome = sm_top.execute()

    # Wait for ctrl-c to stop the application
    rospy.spin()
    sis.stop()


if __name__ == '__main__':
    main()

"""
说明
1.class：一个class为一个状态，execute()为响应时调用的函数

2.StateMachine：状态机，通过add()添加状态，函数说明：
    smach.StateMachine.add(name, class, transitions,remapping)
    
        name，状态机名字
        class，状态机类
        transitions，状态机状态跳转
        remapping，状态机的输入输出映射到整体空间中,用于数据传递，input_keys为输入,output_keys为输出

3.IntrospectionServer：内部检测服务器，用于观察状态机状态
    IntrospectionServer(name, StateMachine,level)
    
        name，观测服务器名字
        StateMachine，状态机
        level，状态机层级
"""
