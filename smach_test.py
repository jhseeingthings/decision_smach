#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import smach
import smach_ros
from smach_ros import MonitorState, IntrospectionServer
import numpy as np
import threading
from multiprocessing.pool import ThreadPool
# import all the msg and srv files

# velocity defined by m/s

MIN_TURNING_RADIUS = 4.5
VEHICLE_WIDTH = 1.86
VEHICLE_LENGTH = 5

def lane_projection(map_x, map_y, map_num, cur_x, cur_y, cur_yaw = 0, type = 0):
    """
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
        dir_diff = math.acos(np.dot(vec_map_dir, vec_yaw_dir) / (np.linalg.norm(vec_map_dir) * np.linalg.norm(vec_yaw_dir)))
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
    after_length += math.sqrt(math.pow(projection_x - map_x[index + 1], 2) + math.pow(projection_y - map_y[index + 1], 2))


    return projection_x, projection_y, index, lateral_distance, dir_diff_signed, before_length, after_length



'''
FSM:
    Input:
        map:
            1.cur_lane_x
            2.cur_lane_y
            3.cur_lane_num
            4.
            
        obstacles:
            1.ID
            2.Type
            3.RoadID
            4.SBegin (Longitudinal length begin)
            5.SEnd (Longitudinal length end)
            6.LBegin (Lateral width begin)
            7.LEnd (lateral width end)
            8.SVelocity (tangential velocity)
            9.LVelocity (normal velocity)
            
        pathBounds:
            1.leftBoundPoints
            2.rightBoundPoints
            
        Pose:
            1.curX
            2.curY
            3.curV
            
        Traffic Lights/Signs:
            1.TrafficLightState
            2.TrafficLightRemainTime
            2.TrafficSignType
            
            
            
map:
    roadCurve
'''

class Obstacle:
    def __init__(self, obstacle_msg):
        self.id = obstacle_msg.id
        self.type = 0
        self.length = 0
        self.width = 0
        # self.height = 0

        self.if_tracked = 0
        self.detectedTime = []
        self.history_center_points = []
        self.history_velocity = []
        self.history_heading = []
        self.obstacle_update(obstacle_msg)

        self.s_begin = 0
        self.s_end = 0
        self.l_begin = 0
        self.l_end = 0
        self.s_velocity = 0
        self.l_velocity = 0
        self.is_moving = 0
        if lane_info != None:
            self.obstacle_projection(lane_info)

        self.on_lane_id = 0
        self.intention = 0
        self.predicted_center_points = []
        self.predicted_headings = []
        self.sub_decision = 0
        self.safe_distance = 0


class TrafficLight:
    def __init__(self):
        self.light_type = 0
        self.color = 0
        self.blinking = False
        self.remain_time = 0
        self.position_x = 0
        self.position_y = 0


class DecisionOutput:
    def __init__(self):
        # The current scenario of the vehicle.
        self.scenario = 0
        # int32 REFPATHFOLLOW = 1
        # int32 EMERGENCYBRAKE = 2
        # int32 PARK = 3

        # Sub-behavior: extracts the obstacles of interest, contains the predicted motion states of the obstacles and the interactive behaviors towards the obstacles.
        self.filteredObstacles = FilteredObstacles()

        # The speed boundary in the present situation.
        self.speedUpperLimit = 0
        self.speedLowerLimit = 0

        # The reference path in the present situation, it can be the target lane or the center line of the boundaries.
        # The target point is the last point in this reference path.
        self.refPath = []

        #Selected parking lot information
        self.SelectedParkingLot = []

#############
# FilteredObstacles.msg
# This message extracts the obstacles that we should notice during the planning period. It provides the predicted motion states of the obstacles and the interactive behaviors towards the obstacles.
class FilteredObstacles:
    def __init__(self):
        # The type of the obstacle.
        # The value could be VEHICLE,  BICYCLE, PEDESTRAIN, CONE, WATERHORSE, RAIL, OTHER, etc.
        self.type = ''

        # The size of the obstacle.
        # The length should be the distance in the direction of heading.
        # The width should be the distance perpendicular to the direction of heading.
        self.length = 0
        self.width = 0

        # The predicted trajectory of the obstacle
        self.predictedCenterPointsTrajectory = []

        # The predicted heading angle of the obstacle.
        self.predictedHeadings = []

        # Is the obstacle static or dynamic?
        # True for a dynamic obstacle.
        # False for a static obstacle.
        self.isMoving = True

        # The reaction towards the obstacle.
        self.decision = 0
        # int32 OVERTAKE=1
        # int32 GIVEWAY=2
        # int32 BYPASS=3

        # A safe distance to be kept from the obstacle.
        self.safeDistance = 0


# 大决策：当前自车的行为
# possible values
PATH_FOLLOW = 1
STOP = 2
EMERGENCY_BRAKE = 3
PARK = 4

# 动静态障碍物
# Is the obstacle static or dynamic?
STATIC = 0
DYNAMIC = 1

# 小决策
# The reaction towards the obstacle
NONE = 0
OVERTAKE = 1
GIVE_WAY = 2
AVOID_COLLISION = 3

global_pose_data = None
lane_info_processed = None
lane_list = {}
obstacles_list = {}
signs_data = None
lights_data = None

decision_output = DecisionOutput()


TIME_ACC = 1
#is computed as the time for accelerating from zero up to speed in the destination lane using the same conservative acceleration.
TIME_DELAY = 1
#is estimated as the maximum system delay.
TIME_SPACE = 1
#is defined as the minimum required temporal spacing between vehicles, where 1 s approximates a vehicle length per 10 mph.

MIN_DISTANCE_GAP = 5 # One car length

def desired_safety_distance(velocity):
    return max(1, velocity * 3.6 / 10) * MIN_DISTANCE_GAP

def user_data_updater(user_data):
    user_data.lane_info_processed = lane_info_processed
    user_data.lane_list = lane_list
    user_data.obstacles_list = obstacles_list
    user_data.signs_data = signs_data
    user_data.lights_data = lights_data


def scenario_definition_decider(scenario, userdata):
    if scenario == '':
        pass


def scenario_transition_decider(scenario, stage, userdata):
    if scenario == 'STARTUP':
        if stage == 'STARTUP_CHECK':

            pass
        elif stage == 'EXECUTE_STARTUP':
            pass
        pass
    elif scenario == 'LANE_FOLLOW':
        if stage == 'IN_LANE_DRIVING':
            pass
        pass
    elif scenario == 'INTERSECTION':
        if stage == 'APPROACH_INTERSECTION':
            pass
        elif stage == 'CREEP_TO_INTERSECTION':
            pass
        elif stage == 'PASS_INTERSECTION':
            pass
        pass
    elif scenario == 'U_TURN':
        if stage == 'CREEP_FOR_OPPORTUNITY':
            pass
        elif stage == 'EXECUTE_U_TURN':
            pass
        pass
    elif scenario == 'PARK':
        if stage == 'APPROACH_PARKING_SPOT':
            pass
        elif stage == 'EXECUTE_PARK':
            pass
        pass
    elif scenario == 'RE_GLOBAL_PLANNING':
        if stage == 'MOVING_FORWARD':
            pass
        elif stage == 'STOP':
            pass
        pass
    elif scenario == 'EMERGENCY_BRAKE':
        if stage == 'MOVING_FORWARD':
            pass
        elif stage == 'STOP':
            pass
        elif stage == 'EMERGENCY_STOP_STANDBY':
            pass
        pass


    pass

def intention_decider():
    pass

def obstacles_prediction():
    pass

def parking_spot_choose_decider():
    pass


def available_lanes_selector(lane_list, pose_data, obstacles_list):
    """

    :return: a set of available lanes
    """
    available_lanes = {}
    for lane_index in lane_list.keys():
        temp_lane = lane_list[lane_index]
        if temp_lane.priority <= 0:
            continue
        points_x, points_y = [], []
        for j in range(len(temp_lane.points)):
            points_x.append(temp_lane.points[j].x)
            points_y.append(temp_lane.points[j].y)
        points_num = len(points_x)

        #计算自车所处的位置，选择自车前方的静态障碍物来考虑。
        vehicle_result = lane_projection(points_x, points_y, points_num,
                                             pose_data.mapX, pose_data.mapY)
        lon_distance_interest = vehicle_result[5] - MIN_TURNING_RADIUS

        lane_scale = np.arange(-temp_lane.width/2, temp_lane.width/2, 0.1)
        lane_occupancy = np.zeros(len(lane_scale))

        for obstacle_index in obstacles_list.keys():
            temp_obstacle = obstacles_list[obstacle_index]
            if temp_obstacle.is_moving == 0:
                lateral_range = []
                for point_index in temp_obstacle.cur_bounding_points:
                    result = lane_projection(points_x, points_y, points_num,
                                             temp_obstacle.cur_bounding_points[point_index][0],
                                             temp_obstacle.cur_bounding_points[point_index][1])
                    # longitudinal condition
                    if result[5] > lon_distance_interest:
                        # lateral condition
                        if abs(result[3]) < temp_lane.width / 2:
                            lateral_range.append(result[3])
                lateral_min = min(lateral_range)
                lateral_max = max(lateral_range)
                for i in range(len(lane_scale)):
                    if lane_scale[i] > lateral_min and lane_scale[i] < lateral_max:
                        lane_occupancy[i] = 1
        max_zeros_length = 0
        temp_length = 0
        for i in range(len(lane_occupancy)):
            if lane_occupancy[i] == 0:
                temp_length += 1
                if temp_length > max_zeros_length:
                    max_zeros_length = temp_length
            else:
                temp_length = 0
        if max_zeros_length * 0.1 > 1.1 * VEHICLE_WIDTH:
            available_lanes[lane_index] = temp_lane
    return available_lanes

def target_lane_choose_decider(lane_info_processed, available_lanes, pose_data):
    """

    :param lane_info_processed:
    :param lane_list:
    :param pose_data:
    :return: target lane id
    """
    target_lane_id = -1
    # 当前没有目标车道或者当前车道优先级不为正,为了切入车道，只需在场景开始时选择一次
    if lane_info_processed.cur_lane_id == -1 or lane_info_processed.cur_priority <= 0:
        min_distance = 1000
        for lane_index in available_lanes.keys():
            temp_lane = available_lanes[lane_index]
            # 暂时先考虑最近的车道
            points_x, points_y = [], []
            for j in range(len(temp_lane.points)):
                points_x.append(temp_lane.points[j].x)
                points_y.append(temp_lane.points[j].y)
            points_num = len(points_x)
            vehicle_result = lane_projection(points_x, points_y, points_num, pose_data.mapX, pose_data.mapY, pose_data.mapHeading)
            if min_distance > vehicle_result[3]:
                min_distance = vehicle_result[3]
                target_lane_id = lane_index

    # 当前处于正常行驶状态，为了提升行驶效率，而选择目标车道


    return target_lane_id

def traffic_rules_decider():
    pass

def speed_limit_decider():
    pass

def path_priority_decider():
    pass

def re_global_planning_decider():
    pass



#########################################
# STARTUP
#########################################
class StartupCheck(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['ready'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data', 'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'])
    def execute(self, user_data):
        # reset the output
        while(1):
            rospy.sleep(1)
            user_data_updater(user_data)
            # check every input
            input_status = 1
            if input_status == 0:
                continue
            # check vehicle status
            vehicle_status = 1
            if vehicle_status == 0:
                continue
            # check the obstacles in the conflicting ares.
            target_lane_choose_decider(user_data.lane_info_processed, user_data.lane_list, user_data.pose_data)
            # time_action =
            # time_required =





class Startup(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        user_data_updater(user_data)
        pass

#########################################
# IN LANE DRIVING
#########################################
class InLaneDriving(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['park', 'intersection', 'merge_and_across', 'need_to_change_lane', 'error'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )
    def execute(self, user_data):
        while(1):
            rospy.sleep(1)
            user_data_updater(user_data)

            # compare the reward value among the surrounding lanes.

            # if the vehicle on the surrounding lanes is about to cut into this lane. decelerate.
            pass

class LaneChangePreparing(smach.State):
    """
    enter this state when there is a need to change lane.
    select the target lane considering mission and obstacles.
    then, choose a target interval on the target lane.
    return to main state when there is no need to change lane.
    """
    def __init__(self):
        smach.State.__init__(self, outcomes = ['cancel_intention', 'ready_to_change_lane'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )
    def execute(self, user_data):
        user_data_updater(user_data)
        pass

class LaneChanging(smach.State):
    """

    """
    def __init__(self):
        smach.State.__init__(self, outcomes = ['lane_change_completed', 'lane_change_cancelled'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )
    def execute(self, user_data):
        user_data_updater(user_data)
        pass

class FindRecoverySolution(smach.State):
    """
    find a solution step by step.
    first, bypass the dynamic obstacles.
    then, rule out priority
    finally, rule out traffic restrictions.
    """
    def __init__(self):
        smach.State.__init__(self, outcomes = ['need_to_change_lane', 'back_to_normal'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )
    def execute(self, user_data):
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
        smach.State.__init__(self, outcomes = ['cancel_intention', 'ready_to_change_lane'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )
    def execute(self, user_data):
        user_data_updater(user_data)
        pass

class LaneChangingErrorRecovery(smach.State):
    """
    When lane change completes, move back to the original lane.
    """
    def __init__(self):
        smach.State.__init__(self, outcomes = ['lane_change_completed', 'lane_change_cancelled'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )
    def execute(self, user_data):
        user_data_updater(user_data)
        pass

#########################################
# INTERSECTION
#########################################
class ApproachIntersection(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['with_lights', 'without_lights'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass


class CreepToIntersectionWithLights(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['enter'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class CreepToIntersectionWithoutLights(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['enter'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class EnterIntersection(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['pass'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class PassIntersection(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

#########################################
# MERGE AND ACROSS
#########################################
class CreepForOpportunity(smach.State):
    """
    consider the target lane which will merge into and the lanes which will across.
    """
    def __init__(self):
        smach.State.__init__(self, outcomes = ['ready'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class ExecuteMerge(smach.State):
    """

    """
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded', 'break'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class YieldBreak(smach.State):
    """

    """
    def __init__(self):
        smach.State.__init__(self, outcomes = ['continue'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

#########################################
# PARKING
#########################################
class DriveAlongLane(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['enter_parking_zone', 'lane_end'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass


class SelectParkingSpot(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['have_empty_spot', 'no_emtpy_spot'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class DriveAndStopInFront(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['finished'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass


class ExecutePark(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded', 'failed'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class AwaitMission(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['continue'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class MarkParkingSpot(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class ReturnToLane(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class ReGlobalPlan(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['continue', 'need_to_turn_around'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class TurnAround(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded', 'failed'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class Reverse(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['okay_to_turn_around'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

#########################################
# EMERGENCY BRAKE
#########################################
class ConditionJudge(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['satisfied'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

class StopImmediately(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

#########################################
# RE-GLOBAL PLANNING
#########################################
class EmergencyBrake(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['brakeOn', 'brakeOff'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):

        pass

class Await(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['continue'],
                             input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                         'lights_data', 'pose_data'],
                             output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                          'lights_data', 'pose_data']
                             )

    def execute(self, user_data):
        pass

def re_global_planning():
    rospy.init_node('greetings_client')
    #   等待有可用的服务"greetings"
    rospy.wait_for_service("greetings")
    try:
        # 定义service客户端，service 名称为 “greetings”，service 类型为 Greeting
        greetings_client = rospy.ServiceProxy("greetings", Greeting)
        # 向server端发送请求,发送的request内容为 name 和 age，其值分别为 "HAN", 20
        # 此处发送的 request 内容与 srv 文件中定义的 request 部分的属性是一致的
        # resp = greetings_client("HAN",20)
        resp = greetings_client.call("HAN", 20)
        rospy.loginfo("Message From server:%s" % resp.feedback)
    except rospy.ServiceException, e:
        rospy.logwarn("Service call failed: %s" % e)


def main():
    rospy.init_node('decision_smach')

    # Create the top level SMACH state machine
    sm_top = smach.StateMachine(outcomes=[])
    sm_top.userdata.lane_info_processed = None
    sm_top.userdata.lane_list = {}
    sm_top.userdata.obstacles_list = {}
    sm_top.userdata.signs_data = None
    sm_top.userdata.lights_data = None
    sm_top.userdata.pose_data = None

    # Open the container
    with sm_top:

        # Create the sub SMACH state machine
        sm_con = smach.Concurrence(outcomes=['outcome5'], default_outcome='outcome5',
                                   input_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                               'lights_data', 'pose_data'],
                                   output_keys=['lane_info_processed', 'lane_list', 'obstacles_list', 'signs_data',
                                                'lights_data', 'pose_data']
                                   )
        # {'outcome5':{'FOO': 'outcome2','BAR': 'outcome1'}}表示 FOO 和 BAR 输出都要满足条件才会输出 outcome5

        # Open the container
        with sm_con:
            sm_con_scenario = smach.StateMachine(outcomes=['outcome4'],
                                                 input_keys=['lane_info_processed', 'lane_list', 'obstacles_list',
                                                             'signs_data', 'lights_data', 'pose_data'],
                                                 output_keys=['lane_info_processed', 'lane_list', 'obstacles_list',
                                                              'signs_data',
                                                              'lights_data', 'pose_data']
                                                 )

            with sm_con_scenario:
                sm_scenario_startup = smach.StateMachine(outcomes=['succeeded'],
                                                         input_keys=['lane_info_processed', 'lane_list',
                                                                     'obstacles_list', 'signs_data', 'lights_data',
                                                                     'pose_data'],
                                                         output_keys=['lane_info_processed', 'lane_list',
                                                                      'obstacles_list', 'signs_data',
                                                                      'lights_data', 'pose_data']
                                                         )
                with sm_scenario_startup:
                    smach.StateMachine.add('STARTUP_CHECK', StartupCheck(), transitions={'ready': 'EXECUTE_STARTUP'})
                    smach.StateMachine.add('EXECUTE_STARTUP', Startup(), transitions={'succeeded': 'succeeded'})
                smach.StateMachine.add('STARTUP', sm_scenario_startup, transitions={'succeeded': 'LANE_FOLLOW'})

                sm_scenario_lane_follow = smach.StateMachine(outcomes=['park', 'intersection', 'merge_and_across'],
                                                             input_keys=['lane_info_processed', 'lane_list',
                                                                         'obstacles_list', 'signs_data', 'lights_data',
                                                                         'pose_data'],
                                                             output_keys=['lane_info_processed', 'lane_list',
                                                                          'obstacles_list', 'signs_data',
                                                                          'lights_data', 'pose_data']
                                                             )
                with sm_scenario_lane_follow:
                    smach.StateMachine.add('IN_LANE_DRIVING', InLaneDriving(), transitions={'park': 'park',
                                                                                            'intersection': 'intersection',
                                                                                            'merge_and_across': 'merge_and_across',
                                                                                            'need_to_change_lane': 'LANE_CHANGE_PREPARING',
                                                                                            'error': 'ERROR_RECOVERY'})
                    smach.StateMachine.add('LANE_CHANGE_PREPARING', LaneChangePreparing(),
                                           transitions={'cancel_intention': 'IN_LANE_DRIVING',
                                                        'ready_to_change_lane': 'LANE_CHANGING'})
                    smach.StateMachine.add('LANE_CHANGING', LaneChanging(),
                                           transitions={'lane_change_completed': 'IN_LANE_DRIVING',
                                                        'lane_change_cancelled': 'LANE_CHANGE_PREPARING'})
                    sm_scenario_lane_follow_error_recovery = smach.StateMachine(outcomes=['back_to_normal'],
                                                                                input_keys=['lane_info_processed',
                                                                                            'lane_list',
                                                                                            'obstacles_list',
                                                                                            'signs_data', 'lights_data',
                                                                                            'pose_data'],
                                                                                output_keys=['lane_info_processed',
                                                                                             'lane_list',
                                                                                             'obstacles_list',
                                                                                             'signs_data',
                                                                                             'lights_data', 'pose_data']
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
                                                                                            'merge_and_across': 'MERGE_AND_ACROSS'})

                sm_scenario_intersection = smach.StateMachine(outcomes=['succeeded'],
                                                              input_keys=['lane_info_processed', 'lane_list',
                                                                          'obstacles_list', 'signs_data', 'lights_data',
                                                                          'pose_data'],
                                                              output_keys=['lane_info_processed', 'lane_list',
                                                                           'obstacles_list', 'signs_data',
                                                                           'lights_data', 'pose_data']
                                                              )
                with sm_scenario_intersection:
                    smach.StateMachine.add('APPROACH_INTERSECTION', ApproachIntersection(),
                                           transitions={'with_lights': 'CREEP_TO_INTERSECTION_WITH_LIGHTS',
                                                        'without_lights': 'CREEP_TO_INTERSECTION_WITHOUT_LIGHTS'})
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

                sm_scenario_merge = smach.StateMachine(outcomes=['succeeded'],
                                                       input_keys=['lane_info_processed', 'lane_list', 'obstacles_list',
                                                                   'signs_data', 'lights_data', 'pose_data'],
                                                       output_keys=['lane_info_processed', 'lane_list',
                                                                    'obstacles_list', 'signs_data',
                                                                    'lights_data', 'pose_data']
                                                       )
                with sm_scenario_merge:
                    smach.StateMachine.add('CREEP_FOR_OPPORTUNITY', CreepForOpportunity(),
                                           transitions={'ready': 'EXECUTE_MERGE'})
                    smach.StateMachine.add('EXECUTE_MERGE', ExecuteMerge(),
                                           transitions={'succeeded': 'succeeded', 'break': 'YIELD_BREAK'})
                    smach.StateMachine.add('YIELD_BREAK', YieldBreak(), transitions={'continue': 'EXECUTE_MERGE'})
                smach.StateMachine.add('MERGE_AND_ACROSS', sm_scenario_merge, transitions={'succeeded': 'LANE_FOLLOW'})

                sm_scenario_park = smach.StateMachine(outcomes=['mission_continue'],
                                                      input_keys=['lane_info_processed', 'lane_list', 'obstacles_list',
                                                                  'signs_data', 'lights_data', 'pose_data'],
                                                      output_keys=['lane_info_processed', 'lane_list', 'obstacles_list',
                                                                   'signs_data',
                                                                   'lights_data', 'pose_data']
                                                      )
                with sm_scenario_park:
                    smach.StateMachine.add('DRIVE_ALONG_LANE', DriveAlongLane(),
                                           transitions={'enter_parking_zone': 'SELECT_PARKING_SPOT',
                                                        'lane_end': 'RE_GLOBAL_PLAN'})
                    smach.StateMachine.add('SELECT_PARKING_SPOT', SelectParkingSpot(),
                                           transitions={'have_empty_spot': 'DRIVE_AND_STOP_IN_FRONT',
                                                        'no_emtpy_spot': 'MARK_PARKING_SPOT'})
                    smach.StateMachine.add('DRIVE_AND_STOP_IN_FRONT', DriveAndStopInFront(),
                                           transitions={'finished': 'EXECUTE_PARK'})
                    smach.StateMachine.add('EXECUTE_PARK', ExecutePark(),
                                           transitions={'succeeded': 'AWAIT_MISSION',
                                                        'failed': 'MARK_PARKING_SPOT'})
                    smach.StateMachine.add('AWAIT_MISSION', AwaitMission(),
                                           transitions={'continue': 'mission_continue'})
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

            # sm_re_global_Planning = smach.StateMachine(outcomes=['succeeded'],
            #                                            input_keys=['lane_info_processed', 'lane_list', 'obstacles_list',
            #                                                        'signs_data', 'lights_data', 'pose_data'],
            #                                            output_keys=['lane_info_processed', 'lane_list',
            #                                                         'obstacles_list', 'signs_data',
            #                                                         'lights_data', 'pose_data']
            #                                            )
            # with sm_re_global_Planning:
            #     smach.StateMachine.add('MOVING_FORWARD', ConditionJudge(), transitions={'satisfied': 'STOP'})
            #     smach.StateMachine.add('STOP', StopImmediately(), transitions={'succeeded': 'MOVING_FORWARD'})
            # smach.Concurrence.add('RE_GLOBAL_PLANNING', sm_re_global_Planning)

            # sm_emergency_brake = smach.StateMachine(outcomes=['succeeded'])
            # with sm_emergency_brake:
            #     smach.StateMachine.add('MOVING_FORWARD', EmergencyBrake(), transitions={'brakeOn': 'STOP'})
            #     smach.StateMachine.add('STOP', StopImmediately(), transitions={'succeeded': 'AWAIT'})
            #     smach.StateMachine.add('AWAIT', StopImmediately(), transitions={'succeeded': 'MOVING_FORWARD'})
            # smach.Concurrence.add('EMERGENCY_BRAKE', sm_emergency_brake)

        smach.StateMachine.add('FINITE_STATE_MACHINE', sm_con, transitions = {'outcome5': 'FINITE_STATE_MACHINE'})

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
————————————————
版权声明：本文为CSDN博主「white_Learner」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/Kalenee/java/article/details/89466999

"""




"""
msgs = Lights()
    for direction,color in zip(directions,colors):  
        #turn detection results into ros message  
        msg = Light()   
        msg.directionIndication=direction
        msg.color = color          
        msgs.lights.append(msg) 

"""