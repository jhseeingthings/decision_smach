# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from local_messages.msg import Road
from local_messages.msg import GlobalPose
from local_messages.msg import Obstacles
from local_messages.msg import Boundaries
from local_messages.msg import Lights
from local_messages.msg import Signs


import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.path import Path

road_data = None
global_pose_data = None
# obstacles_data = None
boundaries_data = None
signs_data = None
lights_data = None

lane_info_processed = None
lane_list = {}

# obstacles_info = None
obstacles_list = {}
cur_lane_list = []
UNCLASSIFIED = False
NOISE = None


WIDTH = 1.86
LENGTH = 4.99
LENGTH_REAR = 1.50


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


def road_callback(road_msg):
    global road_data, lane_list
    road_data = road_msg

    lane_list = {} # {'id':'lane'}
    for k in range(len(road_data.lanes)):
        lane_list[road_data.lanes[k].id] = road_data.lanes[k]
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def global_pose_callback(global_pose_msg):
    global global_pose_data, lane_info_processed
    global_pose_data = global_pose_msg
    # process lane information when the global pose updates
    if road_data != None:
        lane_info_processed = LaneInfoUpdate()

def boundaries_callback(boundaries_msg):
    global boundaries_data
    boundaries_data = boundaries_msg

def lights_callback(lights_msg):
    global lights_data
    lights_data = lights_msg

def signs_callback(signs_msg):
    global signs_data
    signs_data = signs_msg

def obstacles_callback(obstacles_msg):
    global obstacles_data, obstacles_list

    # record road data of the current moment.
    temp_lane_info = road_data
    # reset the if_tracked tag
    for k in obstacles_list.keys():
        obstacles_list[k].if_tracked = 0

    for i in range(len(obstacles_msg.obstacles)):
        # update old obstacles
        if obstacles_msg.obstacles[i].id in obstacles_list.keys():
            obstacles_list[obstacles_msg.obstacles[i].id].obstacle_update(obstacles_msg.obstacles[i], temp_lane_info)
        # generate new obstacle
        if obstacles_msg.obstacles[i].id not in obstacles_list.keys():
            temp_obstacle = Obstacle(obstacles_msg.obstacles[i], temp_lane_info)
            obstacles_list[obstacles_msg.obstacles[i].id] = temp_obstacle

    for k in obstacles_list.keys():
        if obstacles_list[k].if_tracked == 0:
            del obstacles_list[k]


def listener():
    # 注意node的名字得独一无二，但是topic的名字得和你想接收的信息的topic一样！
    rospy.init_node('listener', anonymous = True)

    # Subscriber函数第一个参数是topic的名称，第二个参数是接受的数据类型，第三个参数是回调函数的名称
    rospy.Subscriber("global_pose", GlobalPose, global_pose_callback)
    rospy.Subscriber("map_road", Road, road_callback)
    rospy.Subscriber("fused_obstacles", Obstacles, obstacles_callback)
    rospy.Subscriber("boundaries", Boundaries, boundaries_callback)
    rospy.Subscriber("traffic_lights", Lights, lights_callback)
    rospy.Subscriber("traffic_signs", Signs, signs_callback)

    # spin() simply keeps python from exiting until this node is stopped
    # 只 spin 有 callback 的语句
    rospy.spin()

#########################
# extract center points from two boundaries.
def getBoundariesCenterPoints(boundary1, boundary2):
    boundaryPointsNumber1 = len(boundary1)
    boundaryPointsNumber2 = len(boundary2)
    midPointsList1 = []
    midPointsList2 = []
    for i in range(boundaryPointsNumber1):
        x1 = boundary1[i][0]
        y1 = boundary1[i][1]
        minDistance = 1000000
        for j in range(boundaryPointsNumber2):
            x2 = boundary2[j][0]
            y2 = boundary2[j][1]
            tempDistance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if (tempDistance < minDistance):
                minDistance = tempDistance
                nearestPointX = x2
                nearestPointY = y2
        midPointX = (x1 + nearestPointX) / 2
        midPointY = (y1 + nearestPointY) / 2
        midPointsList1.append([midPointX, midPointY])
    for i in range(boundaryPointsNumber2):
        x2 = boundary2[i][0]
        y2 = boundary2[i][1]
        minDistance = 1000000
        for j in range(boundaryPointsNumber1):
            x1 = boundary1[j][0]
            y1 = boundary1[j][1]
            tempDistance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if (tempDistance < minDistance):
                minDistance = tempDistance
                nearestPointX = x1
                nearestPointY = y1
        midPointX = (x2 + nearestPointX) / 2
        midPointY = (y2 + nearestPointY) / 2
        midPointsList2.append([midPointX, midPointY])
    return midPointsList1, midPointsList2


# find the nearest point ahead
def getClosestPoint(curPointPositionX, curPointPositionY, curPointPositionYaw, pointList):
    minDistance = 100000
    closestMidPointX = -1
    closestMidPointY = -1
    for i in range(len(pointList)):
        vehicle2MidPoint = np.array([pointList[i][0] - curPointPositionX, pointList[i][1] - curPointPositionY])
        vec_yaw = np.array([math.cos(curPointPositionYaw), math.sin(curPointPositionYaw)])
        cosAngel = np.dot(vehicle2MidPoint, vec_yaw) / np.linalg.norm(vehicle2MidPoint) / np.linalg.norm(vec_yaw)
        tempDistance = math.sqrt((pointList[i][0] - curPointPositionX) ** 2 + (pointList[i][1] - curPointPositionY) ** 2)
        if tempDistance < minDistance and cosAngel > 0.5:
            minDistance = tempDistance
            closestMidPointX = pointList[i][0]
            closestMidPointY = pointList[i][1]
    return closestMidPointX, closestMidPointY


# remove duplicated points, rearrange points with direction and distance
def getBoundariesCenterLine(midPointsList1, midPointsList2, curX, curY, curYaw):
    preMassMidPointsList = []
    for i in range(len(midPointsList1)):
        preMassMidPointsList.append(midPointsList1[i])
    for i in range(len(midPointsList2)):
        preMassMidPointsList.append(midPointsList2[i])
    massMidPointsList = []
    # 去除重复点
    for i in range(len(preMassMidPointsList)):
        if preMassMidPointsList[i] not in massMidPointsList:
            massMidPointsList.append(preMassMidPointsList[i])
        else:
            pass
    centerLine = []
    curPointPositionX = curX
    curPointPositionY = curY
    curPointPositionYaw = curYaw
    pointList = []
    print(massMidPointsList)
    for i in range(len(massMidPointsList)):
        pointList.append(massMidPointsList[i])
    while (1):
        closestMidPointX, closestMidPointY = getClosestPoint(curPointPositionX, curPointPositionY, curPointPositionYaw,
                                                             pointList)
        if closestMidPointX == -1 and closestMidPointY == -1:
            break
        centerLine.append([closestMidPointX, closestMidPointY])
        pointList.remove([closestMidPointX, closestMidPointY])
        curPointPositionYaw = math.atan2(closestMidPointY - curPointPositionY, closestMidPointX - curPointPositionX)
        curPointPositionX = closestMidPointX
        curPointPositionY = closestMidPointY

    return centerLine


# update lane information, triggered when the global pose is updated.
class LaneInfoUpdate:
    def __init__(self):
        self.lanes = road_data.lanes
        self.offset = []
        self.dir_diff = []
        self.before_length = [] # 记录对于每一段 lane 已经行驶过的长度
        self.after_length = [] # 记录对于每一段 lane 没有行驶过的长度
        self.points_x = []
        self.points_y = []
        self.points_num = 0

        self.cur_lane_x = []
        self.cur_lane_y = []
        self.cur_lane_num = 0
        self.cur_lane_id = 0
        self.left_lane_id = 0
        self.right_lane_id = 0
        self.cur_lane_width = 0
        self.left_lane_width = 0
        self.right_lane_width = 0
        self.next_stop_type = 0
        self.dist_to_next_stop = 0
        self.dist_to_next_road = 0
        self.cur_turn_type = 0
        self.next_turn_type = 0
        self.can_change_left = 0
        self.can_change_right = 0
        self.speed_upper_limit = 0
        self.speed_lower_limit = 0
        self.lead_to_ids = []
        self.lead_to_preferred = []
        self.cur_preferred = 0
        self.left_preferred = []
        self.right_preferred = []

        self.lane_of_interest = []

        self.lane_message_process()

    def lane_message_process(self):
        for i in range(len(self.lanes)):
            self.points_x = []
            self.points_y = []
            for j in range(len(self.lanes[i].points)):
                self.points_x.append(self.lanes[i].points[j].x)
                self.points_y.append(self.lanes[i].points[j].y)
            self.points_num = len(self.points_x)
            result = lane_projection(self.points_x, self.points_y, self.points_num, global_pose_data.mapX, global_pose_data.mapY, global_pose_data.mapHeading)
            self.offset.append(result[3])
            self.dir_diff.append(result[4])
            self.before_length.append(result[5])
            self.after_length.append(result[6])

        # choose current lane id and index
        # DIR_THRESHOLD = 120.0 / 180.0 * 3.14159265
        OFFSET_THRESHOLD = 1.0
        min_offset = 2.0
        cur_lane_index = -1
        count = 0
        preferred_index_set = []
        preferred_id_set = []


        # 选择当前车道(找全局规划)
        # 先找距离处于较小范围内，优先级较高的第一条车道
        # 如果找不到，找距离最小的一条车道
        for i in range(len(self.lanes)):
            abs_offset = abs(self.offset[i])
            # 附近车道，距离最小，方向偏差小，优先级高（2），不是车道末段
            if self.lanes[i].relation == 1 and abs_offset < OFFSET_THRESHOLD \
                    and self.lanes[i].preferred == 2 \
                    and self.after_length[i] > 2:
                count += 1
                preferred_index_set.append(i)
                preferred_id_set.append(self.lanes[i].id)
        if count == 0:
            for i in range(len(self.lanes)):
                abs_offset = abs(self.offset[i])
                # 附近车道，距离最小，方向偏差小，优先级为 1，不是车道末段
                if self.lanes[i].relation == 1 and abs_offset < OFFSET_THRESHOLD \
                        and self.lanes[i].preferred == 1 \
                        and self.after_length[i] > 2:
                    count += 1
                    preferred_index_set.append(i)
                    preferred_id_set.append(self.lanes[i].id)
        if count == 0:
            for i in range(len(self.lanes)):
                abs_offset = abs(self.offset[i])
                #
                if self.lanes[i].relation == 1 and abs_offset < min_offset \
                        and self.after_length[i] > 2:
                    count = 1
                    min_offset = abs_offset
                    preferred_index_set[0] = i  # 只有一条
                    preferred_id_set[0] = self.lanes[i].id

        # for i in range(len(self.lanes)):
        #     abs_offset = abs(self.offset[i])
        #     abs_dir_diff = abs(self.dir_diff[i])
        #     # 附近车道，距离最小，方向偏差小，优先级高（2），不是车道末段
        #     if self.lanes[i].relation == 1 and abs_offset < OFFSET_THRESHOLD \
        #             and abs_dir_diff < DIR_THRESHOLD and self.lanes[i].preferred == 2 \
        #             and self.after_length[i] > 5:
        #         count += 1
        #         preferred_index_set.append(i)
        #         preferred_id_set.append(self.lanes[i].id)
        # if count == 0:
        #     for i in range(len(self.lanes)):
        #         abs_offset = abs(self.offset[i])
        #         abs_dir_diff = abs(self.dir_diff[i])
        #         # 附近车道，距离最小，方向偏差小，优先级为 1，不是车道末段
        #         if self.lanes[i].relation == 1 and abs_offset < OFFSET_THRESHOLD \
        #                 and abs_dir_diff < DIR_THRESHOLD and self.lanes[i].preferred == 1 \
        #                 and self.after_length[i] > 5:
        #             count += 1
        #             preferred_index_set.append(i)
        #             preferred_id_set.append(self.lanes[i].id)
        # if count == 0:
        #     for i in range(len(self.lanes)):
        #         abs_offset = abs(self.offset[i])
        #         abs_dir_diff = abs(self.dir_diff[i])
        #         #
        #         if self.lanes[i].relation == 1 and abs_offset < min_offset \
        #                 and abs_dir_diff < DIR_THRESHOLD and self.lanes[i].preferred > 0 \
        #                 and self.after_length[i] > 5:
        #             count = 1
        #             min_offset = abs_offset
        #             preferred_index_set[0] = i  # 只有一条
        #             preferred_id_set[0] = self.lanes[i].id

        # 对于多条符合要求的 lanes ，选一条(id 最小的一条，来保证连续性）

        if count != 0:
            min_id = 10000
            min_id_index = -1
            for i in range(count):
                if preferred_id_set[i] < min_id:
                    min_id = preferred_id_set[i]
                    min_id_index = preferred_index_set[i]
            cur_lane_index = min_id_index

        if cur_lane_index != -1:
            temp_lead_to_id = -1
            lead_to_index = 0
            for i in range(len(self.lanes[cur_lane_index].leadToIds)):
                for j in range(len(self.lanes)):
                    if self.lanes[j].id == self.lanes[cur_lane_index].leadToIds[i] \
                            and self.lanes[j].preferred == 2:
                        temp_lead_to_id = self.lanes[j].id
                        lead_to_index = j
                        break
                if temp_lead_to_id != -1:
                    break

            if temp_lead_to_id == -1:
                for i in range(len(self.lanes[cur_lane_index].leadToIds)):
                    for j in range(len(self.lanes)):
                        if self.lanes[j].id == self.lanes[cur_lane_index].leadToIds[i] \
                                and self.lanes[j].preferred == 1:
                            temp_lead_to_id = self.lanes[j].id
                            lead_to_index = j
                            break
                    if temp_lead_to_id != -1:
                        break

            # generate and merge current lane
            for i in range(len(self.lanes[cur_lane_index].points)):
                self.cur_lane_x.append(self.lanes[cur_lane_index].points[i].x)
                self.cur_lane_y.append(self.lanes[cur_lane_index].points[i].y)
            # merge !!!
            if self.after_length[cur_lane_index] < 10 and temp_lead_to_id != -1:
                for j in range(len(self.lanes[lead_to_index].points)):
                    self.cur_lane_x.append(self.lanes[lead_to_index].points[j].x)
                    self.cur_lane_y.append(self.lanes[lead_to_index].points[j].y)

            self.cur_lane_num = len(self.cur_lane_x)
            self.cur_lane_id = self.lanes[cur_lane_index].id
            self.cur_lane_width = self.lanes[cur_lane_index].width
            # 需要查到如果没有左右车道时 id 是什么
            left_lane_index = -1
            right_lane_index = -1
            for i in range(len(self.lanes)):
                if self.lanes[i].id == self.lanes[cur_lane_index].leftLaneId:
                    left_lane_index = i
                elif self.lanes[i].id == self.lanes[cur_lane_index].rightLaneId:
                    right_lane_index = i
            if left_lane_index != -1:
                self.left_lane_id = self.lanes[left_lane_index].id
                self.left_lane_width = self.lanes[left_lane_index].width
            else:
                self.left_lane_id = -1
            if right_lane_index != -1:
                self.right_lane_id = self.lanes[right_lane_index].id
                self.right_lane_width = self.lanes[right_lane_index].width
            else:
                self.right_lane_id = -1

            if self.lanes[cur_lane_index].stopType != 0:
                stop_line_x = self.lanes[cur_lane_index].nextStop.x
                stop_line_y = self.lanes[cur_lane_index].nextStop.y
                self.dist_to_next_stop = math.sqrt(math.pow(stop_line_x - global_pose_data.mapX, 2) + pow(stop_line_y - global_pose_data.mapY, 2))
                self.next_stop_type = self.lanes[cur_lane_index].stopType

            elif self.lanes[lead_to_index].stopType != 0 and temp_lead_to_id != 0:
                stop_line_x = self.lanes[lead_to_index].nextStop.x
                stop_line_y = self.lanes[lead_to_index].nextStop.y
                dist_section_1 = self.after_length[cur_lane_index]
                dist_section_2 = math.sqrt(math.pow(stop_line_x - self.lanes[cur_lane_index].points[-1].x, 2) + math.pow(stop_line_y - self.lanes[cur_lane_index].points[-1].y, 2))
                self.dist_to_next_stop = dist_section_1 + dist_section_2
                self.next_stop_type = self.lanes[lead_to_index].stopType

            self.dist_to_next_road = self.after_length[cur_lane_index]
            self.cur_turn_type = self.lanes[cur_lane_index].turn
            if temp_lead_to_id != 0:
                self.next_turn_type = self.lanes[lead_to_index].turn
            self.can_change_left = self.lanes[cur_lane_index].canChangeLeft
            self.can_change_right = self.lanes[cur_lane_index].canChangeRight
            self.speed_upper_limit = self.lanes[cur_lane_index].speedUpperLimit
            self.speed_lower_limit = self.lanes[cur_lane_index].speedLowerLimit


            self.cur_preferred = self.lanes[cur_lane_index].preferred

            temp_index = cur_lane_index
            for i in range(10):
                temp_left_id = self.lanes[temp_index].leftLaneId
                for j in range(len(self.lanes)):
                    if self.lanes[j].id == temp_left_id:
                        self.left_preferred.append(self.lanes[j].preferred)
                        temp_index = j
                        break
                if j == len(self.lanes):
                    break

            temp_index = cur_lane_index
            for i in range(10):
                temp_right_id = self.lanes[temp_index].rightLaneId
                for j in range(len(self.lanes)):
                    if self.lanes[j].id == temp_right_id:
                        self.right_preferred.append(self.lanes[j].preferred)
                        temp_index = j
                        break
                if j == len(self.lanes):
                    break

            self.lead_to_preferred = []

            for i in range(len(self.lanes[cur_lane_index].leadToIds)):
                self.lead_to_ids.append(self.lanes[cur_lane_index].leadToIds[i])
                for j in range(len(self.lanes)):
                    if self.lanes[cur_lane_index].leadToIds[i] == self.lanes[j].id:
                        self.lead_to_preferred.append(self.lanes[j].preferred)


            if(self.cur_lane_id != cur_lane_list[-1]):
                cur_lane_list.append(self.cur_lane_id)

            vehicle_projection = lane_projection(road_data.cur_lane_x, road_data.cur_lane_y, road_data.cur_lane_num, global_pose_data.mapX, global_pose_data.mapY, global_pose_data.mapHeading)

        else:
            self.cur_lane_id = -1

class Obstacle:
    def __init__(self, obstacle_msg, cur_lane_info):
        self.id = obstacle_msg.id
        self.type = 0
        self.length = 0
        self.width = 0
        self.height = 0
        self.cur_bounding_points = []
        self.cur_velocity = 0
        self.is_moving = 0

        # 时序问题，边界障碍物提取好后，去判断用什么车道作为当前车道，选好当前车道后，做规则障碍物向车道的投影
        self.if_tracked = 0
        self.detected_time = []
        self.history_center_points = []
        self.history_velocity = []
        self.history_heading = []
        self.obstacle_update(obstacle_msg, cur_lane_info)

        self.s_begin = 0
        self.s_end = 0
        self.l_begin = 0
        self.l_end = 0
        self.s_velocity = 0
        self.l_velocity = 0

        self.on_lane_id = 0
        self.intention = 0
        self.predicted_center_points = []
        self.predicted_headings = []
        self.sub_decision = 0
        self.safe_distance = 0

    # update obstacles information, record the history movements of the obstacles.
    def obstacle_update(self, obstacle_msg, cur_lane_info):
        self.type = obstacle_msg.type
        self.if_tracked = 1
        self.cur_bounding_points = [[obstacle_msg.points[i].x, obstacle_msg.points[i].y] for i in range(len(obstacle_msg.points))]
        # like this: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 9]]
        self.cur_velocity = math.sqrt(math.pow(obstacle_msg.velocity.x, 2) + math.pow(obstacle_msg.velocity.y, 2))
        if self.cur_velocity > 0.1:
            self.is_moving = 1
        else:
            self.is_moving = 0

        # record history trajectory for regular obstacles.
        if self.type == 'VEHICLE' or self.type == 'PEDESTRIAN' or self.type == 'BICYCLE':
            self.detected_time.append(obstacle_msg.detectedTime)
            self.history_velocity.append([obstacle_msg.velocity.x, obstacle_msg.velocity.y, obstacle_msg.velocity.z])
            cur_heading = math.atan(obstacle_msg.velocity.y / obstacle_msg.velocity.x)
            self.history_heading.append(cur_heading)

            # calculate center point
            center_point_x = np.mean([obstacle_msg.points[i].x for i in range(len(obstacle_msg.points))])
            center_point_y = np.mean([obstacle_msg.points[i].y for i in range(len(obstacle_msg.points))])
            self.history_center_points.append([center_point_x, center_point_y])

            project_heading = []
            project_lateral = []
            vec_heading = np.array([obstacle_msg.velocity.x, obstacle_msg.velocity.y])
            vec_lateral = np.array([-obstacle_msg.velocity.y, obstacle_msg.velocity.x])
            for i in range(len(obstacle_msg.points)):
                vec_point = np.array([obstacle_msg.points[i].x - center_point_x, obstacle_msg.points[i].y - center_point_y])
                project_heading.append(np.dot(vec_heading, vec_point) / np.linalg.norm(vec_heading))
                project_lateral.append(np.dot(vec_lateral, vec_point) / np.linalg.norm(vec_lateral))

            self.length = max(project_heading) - min(project_heading)
            self.width = max(project_lateral) - min(project_lateral)
        # for other obstacles, update new information.
        else:
            self.detected_time.clear()
            self.history_center_points.clear()
            self.history_velocity.clear()
            self.history_heading.clear()
            self.detected_time.append(obstacle_msg.detectedTime)

        # project the obstacle to the current lane.
        if cur_lane_info != None:
            self.obstacle_projection(cur_lane_info)

    # project obstacles onto the lane
    def obstacle_projection(self, lane_info):
        s_range = []
        l_range = []
        for i in range(len(self.cur_bounding_points)):
            result = lane_projection(lane_info.cur_lane_x, lane_info.cur_lane_y, lane_info.cur_lane_num, self.cur_bounding_points[i][0], self.cur_bounding_points[i][1])
            # L: result[3], S: result[5]
            s_range.append(result[5])
            l_range.append(result[3])
        self.s_begin = min(s_range)
        self.s_end = max(s_range)
        self.l_begin = min(l_range)
        self.l_end = max(l_range)
        if self.is_moving:
            result_center = lane_projection(lane_info.cur_lane_x, lane_info.cur_lane_y, lane_info.cur_lane_num, self.history_center_points[-1][0], self.history_center_points[-1][1], self.history_heading[-1])
            # direction difference : result[3]
            self.s_velocity = math.cos(result_center[4]) * self.velocity
            self.l_velocity = math.sin(result_center[4]) * self.velocity

    # predict the future trajectory of the obstacle
    def obstacle_prediction(self):
        pass

# int obstacle_lane_distance(local_messages::Obstacle obstacle, local_messages::Lane lane, local_messages::GlobalPose pose, float &distance , vector<float> &obswidth,int &outputflag);

if __name__ == '__main__':
    listener()




# class DBSCAN:
#     def __init__(self, m, eps, min_points):
#         self.cluster_result = self.dbscan(m, eps, min_points)
# 
#     def _dist(self, p, q):
#         return math.sqrt(np.power(p - q, 2).sum())
# 
#     def _eps_neighborhood(self, p, q, eps):
#         return self._dist(p, q) < eps
# 
#     def _region_query(self, m, point_id, eps):
#         n_points = m.shape[1]
#         seeds = []
#         for i in range(0, n_points):
#             if self._eps_neighborhood(m[:, point_id], m[:, i], eps):
#                 seeds.append(i)
#         return seeds
# 
#     def _expand_cluster(self, m, classifications, point_id, cluster_id, eps, min_points):
#         seeds = self._region_query(m, point_id, eps)
#         if len(seeds) < min_points:
#             classifications[point_id] = NOISE
#             return False
#         else:
#             # classifications[point_id] = cluster_id
#             for seed_id in seeds:
#                 classifications[seed_id] = cluster_id
#             while len(seeds) > 0:
#                 current_point = seeds[0]
#                 print(seeds)
#                 results = self._region_query(m, current_point, eps)
#                 if len(results) >= min_points:
#                     for i in range(0, len(results)):
#                         result_point = results[i]
#                         if classifications[result_point] == UNCLASSIFIED or \
#                            classifications[result_point] == NOISE:
#                             if classifications[result_point] == UNCLASSIFIED:
#                                 seeds.append(result_point)
#                             classifications[result_point] = cluster_id
#                 seeds = seeds[1:]
#             return True
# 
#     def dbscan(self, m, eps, min_points):
#         """Implementation of Density Based Spatial Clustering of Applications with Noise
#         See https://en.wikipedia.org/wiki/DBSCAN
# 
#         scikit-learn probably has a better implementation
# 
#         Uses Euclidean Distance as the measure
# 
#         Inputs:
#         m - A matrix whose columns are feature vectors
#         eps - Maximum distance two points can be to be regionally related
#         min_points - The minimum number of points to make a cluster
# 
#         Outputs:
#         An array with either a cluster id number or dbscan.NOISE (None) for each
#         column vector in m.
#         """
#         cluster_id = 1
#         n_points = m.shape[1]
#         classifications = [UNCLASSIFIED] * n_points
#         for point_id in range(0, n_points):
#             point = m[:, point_id]
#             if classifications[point_id] == UNCLASSIFIED: # 对于没有分过类的点
#                 if self._expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
#                     cluster_id = cluster_id + 1
# 
#         index_outer = []
#         cluster_number = max(classifications)
#         index_outer.append(cluster_number)
#         for i in range(1, cluster_number + 1):
#             index_inner = []
#             for j in range(len(classifications)):
#                 if classifications[j] == i:
#                     index_inner.append(j)
#             index_outer.append(index_inner)
# 
#         return classifications, index_outer


#
# m = np.matrix('1 0.8 3.7 1.2 3.9 3.6 10 6; 1.1 1 4 0.8 3.9 4.1 10 7')
#     eps = 0.3
#     min_points = 1
#     print(m.shape[0], m.shape[1])
#     cluster_result = DBSCAN(m, eps, min_points)



# # msg 的封装
# msgs = Lights()
# for direction,color in zip(directions,colors):
#         #turn detection results into ros message
#         msg = Light()
#         msg.directionIndication=direction
#         msg.color = color
#         msgs.lights.append(msg)
#     return msgs,traj_image