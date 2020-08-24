#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
作者:Jairus Chan
程序:多项式曲线拟合算法
'''
import matplotlib.pyplot as plt
import math
import numpy as np


PREDICTION_DURATION = 5
PREDICTION_PERIOD = 0.1


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
        self.lead_to_priority = []
        self.cur_priority = 0
        self.left_priority = []
        self.right_priority = []

        self.lane_of_interest = []

lane_info = LaneInfoUpdate()

def compute_mean(nums, start, end):
    sum = 0
    count = 0
    stop = min(end + 1, len(nums))
    print(stop)
    for i in range(start, stop, 1):
        sum += nums[i]
        count += 1
    print(count)
    if count == 0:
        mean = 0
    else:
        mean  = sum / count
    return mean

def curve_fitting(points_x, points_y, points_num, order):
    list_X = []
    for xx in points_x:
        list_line = []
        for i in range(0, order + 1):
            list_line.append(xx**i)
        list_X.append(list_line)
    mat_X = np.reshape(list_X, (points_num, order + 1))

    mat_Y = np.reshape(points_y, (points_num, 1))

    mat_W = np.zeros((points_num, points_num))
    for i in range(points_num):
        mat_W[i, i] = 10000 - i * 9900 / order
        # mat_W[i, i] = 1/mat_W[i,i]
        # mat_W[i, i] = i + 1

    mat_M = np.dot(np.transpose(mat_X), mat_W)
    fitting_result = np.dot(np.dot(np.linalg.inv(np.dot(mat_M, mat_X)), mat_M), mat_Y)
    # temp_result = np.linalg.inv(np.dot(np.transpose(mat_X), mat_X))
    # fitting_result = np.dot(np.dot(temp_result, np.transpose(mat_X)), mat_Y)
    return fitting_result


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

        self.around_lanes = {}

        # 存储每一个时刻运动信息到当前所在车道的投影信息
        self.s_velocity = []
        self.l_velocity = []
        self.dir_diff = []
        self.lane_lateral_diff = []
        self.history_lane_ids = []
        self.s_record = []

        self.cur_lane_id = 0

        self.target_lane_id = 0
        self.next_lane_id = 0
        self.intention = 0 # 0 for free move, 1 for lane keeping , 2 for lane change left, 3 for lane change right. 4 for passing zebra crossing.
        self.predicted_center_points = []
        self.predicted_headings = []

        self.sub_decision = 0
        self.safe_distance = 0

    # update obstacles information, record the history movements of the obstacles.
    def obstacle_update(self, obstacle_msg, cur_lane_info):
        self.around_lanes = cur_lane_info
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

        # project the obstacle to the lanes, and select current lane.
        if cur_lane_info != None:
            self.obstacle_projection(cur_lane_info, self.is_moving)

    # project the obstacle to the lanes, and select current lane.
    def obstacle_projection(self, lane_info, moving_flag):

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
            if last_lane_id in lane_info.keys():
                cur_lane = lane_info[last_lane_id]
                points_x, points_y = [], []
                for j in range(len(cur_lane.points)):
                    points_x.append(cur_lane.points[j].x)
                    points_y.append(cur_lane.points[j].y)
                points_num = len(points_x)
                result = lane_projection(points_x, points_y, points_num, self.history_center_points[-1][0], self.history_center_points[-1][1],
                                            self.history_heading[-1])
                temp_distance = result[3]
                temp_direction_diff = result[4]
                if abs(temp_distance) < distance_threshold and abs(temp_direction_diff) < direction_threshold:
                    self.cur_lane_id = last_lane_id
                    lane_found_flag = True
        # target lane
        if lane_found_flag == False and last_target_lane_id != 0:
            if last_target_lane_id in lane_info.keys():
                cur_lane = lane_info[last_target_lane_id]
                points_x, points_y = [], []
                for j in range(len(cur_lane.points)):
                    points_x.append(cur_lane.points[j].x)
                    points_y.append(cur_lane.points[j].y)
                points_num = len(points_x)
                result = lane_projection(points_x, points_y, points_num, self.history_center_points[-1][0], self.history_center_points[-1][1],
                                            self.history_heading[-1])
                temp_distance = result[3]
                temp_direction_diff = result[4]
                if abs(temp_distance) < distance_threshold and abs(temp_direction_diff) < direction_threshold:
                    self.cur_lane_id = last_target_lane_id
                    lane_found_flag = True
        # if failed, find the current lane.
        if lane_found_flag == False:
            min_distance = 1000
            for k in lane_info.keys():
                cur_lane = lane_info[k]
                points_x, points_y = [], []
                for j in range(len(cur_lane.points)):
                    points_x.append(cur_lane.points[j].x)
                    points_y.append(cur_lane.points[j].y)
                points_num = len(points_x)
                result = lane_projection(points_x, points_y, points_num, self.history_center_points[-1][0],
                                         self.history_center_points[-1][1],
                                         self.history_heading[-1])
                temp_distance = result[3]
                temp_direction_diff = result[4]
                if abs(temp_distance) < distance_threshold and abs(temp_direction_diff) < direction_threshold and abs(temp_distance) < min_distance:
                    min_distance = abs(temp_distance)
                    this_lane_id = k
            if this_lane_id != 0:
                self.cur_lane_id = this_lane_id
                lane_found_flag = True

        if lane_found_flag == False:
            self.cur_lane_id = 0

        # if the current lane found, project history movement to the current lane.
        if lane_found_flag:
            self.lane_lateral_diff = []
            self.dir_diff = []
            self.history_lane_ids = []
            self.s_velocity = []
            self.l_velocity = []
            cur_lane = lane_info[self.cur_lane_id]
            points_x, points_y = [], []
            for j in range(len(cur_lane.points)):
                points_x.append(cur_lane.points[j].x)
                points_y.append(cur_lane.points[j].y)
            points_num = len(points_x)
            for i in range(len(self.history_center_points)):
                result = lane_projection(points_x, points_y, points_num, self.history_center_points[i][0],
                                         self.history_center_points[i][1],
                                         self.history_heading[i])
                self.lane_lateral_diff.append(result[3])
                self.dir_diff.append(result[4])
                self.s_record.append(result[5])
                self.history_lane_ids.append(self.cur_lane_id)
                self.s_velocity.append(math.cos(result[4]) * self.history_velocity[i])
                self.l_velocity.append(math.sin(result[4]) * self.history_velocity[i])

        else:
            pass


        # s_range = []
        # l_range = []
        # for i in range(len(self.cur_bounding_points)):
        #     result = lane_projection(lane_info.cur_lane_x, lane_info.cur_lane_y, lane_info.cur_lane_num,
        #                              self.cur_bounding_points[i][0], self.cur_bounding_points[i][1])
        #     # L: result[3], S: result[5]
        #     s_range.append(result[5])
        #     l_range.append(result[3])
        # self.s_begin = min(s_range)
        # self.s_end = max(s_range)
        # self.l_begin = min(l_range)
        # self.l_end = max(l_range)
        # if self.is_moving:
        #     result_center = lane_projection(lane_info.cur_lane_x, lane_info.cur_lane_y, lane_info.cur_lane_num,
        #                                     self.history_center_points[-1][0], self.history_center_points[-1][1],
        #                                     self.history_heading[-1])
        #     # direction difference : result[3]
        #     self.s_velocity = math.cos(result_center[4]) * self.cur_velocity
        #     self.l_velocity = math.sin(result_center[4]) * self.cur_velocity

    # predict the future intention of the obstacle
    def obstacle_intention_prediction(self):

        if self.type == "VEHICLE" and "BICYCLE":
            if self.cur_lane_id != 0:
                if self.around_lanes[self.cur_lane_id].turn == 0:
                    can_change_left_flag = self.around_lanes[self.cur_lane_id].canChangeLeft
                    can_change_right_flag = self.around_lanes[self.cur_lane_id].canChangeRight
                    lane_lateral_diff_cur = self.lane_lateral_diff[-1]
                    if len(self.lane_lateral_diff) > 10:
                        lane_lateral_diff_mean = sum(self.lane_lateral_diff[len(self.lane_lateral_diff)-11:-1]) / 10
                    elif len(self.lane_lateral_diff) > 1:
                        lane_lateral_diff_mean = sum(self.lane_lateral_diff[0:-1]) / (len(self.lane_lateral_diff)-1)
                    else:
                        lane_lateral_diff_mean = lane_lateral_diff_cur
                    if lane_lateral_diff_cur > lane_lateral_diff_mean and lane_lateral_diff_cur > 0 and self.l_velocity > 0 and can_change_left_flag == 1:
                        self.intention = 2
                        self.target_lane_id = self.around_lanes[self.cur_lane_id].leftLaneId
                    elif lane_lateral_diff_cur < lane_lateral_diff_mean and lane_lateral_diff_cur < 0 and self.l_velocity < 0 and can_change_right_flag == 1:
                        self.intention = 3
                        self.target_lane_id = self.around_lanes[self.cur_lane_id].rightLaneId
                    else:
                        self.intention = 1
                        self.target_lane_id = self.cur_lane_id
                else:
                    # intersection -- lane keeping
                    self.intention = 1
                    self.target_lane_id = self.cur_lane_id
                # select next lane id
                lead_to_ids = self.around_lanes[self.target_lane_id].leadToIds
                vec_end_dir = np.array([self.around_lanes[self.target_lane_id].points[-1].y - self.around_lanes[self.target_lane_id].points[-3].y,
                                          self.around_lanes[self.target_lane_id].points[-1].x - self.around_lanes[self.target_lane_id].points[-3].x])
                min_dir_diff = 100
                for i in range(len(lead_to_ids)):
                    if lead_to_ids[i] in self.around_lanes.keys():
                        vec_start_dir = np.array([self.around_lanes[lead_to_ids[i]].points[3].y - self.around_lanes[lead_to_ids[i]].points[0].y,
                                          self.around_lanes[lead_to_ids[i]].points[3].x - self.around_lanes[lead_to_ids[i]].points[0].x])
                        dir_diff = math.cos(np.dot(vec_end_dir, vec_start_dir) / (np.linalg.norm(vec_end_dir) * np.linalg.norm(vec_start_dir)))
                        if dir_diff < min_dir_diff:
                            min_dir_diff = dir_diff
                            self.next_lane_id = lead_to_ids[i]

            else:
                # free move predictor
                pass

        elif self.type == "PEDESTRIAN":
            pass

    # predict the future intention of the obstacle
    def obstacle_trajectory_prediction(self):


        if self.target_lane_id in lane_info.keys():
            cur_lane = lane_info[self.target_lane_id]
            points_x, points_y = [], []
            for j in range(len(cur_lane.points)):
                points_x.append(cur_lane.points[j].x)
                points_y.append(cur_lane.points[j].y)
            points_num = len(points_x)
            result = lane_projection(points_x, points_y, points_num, self.history_center_points[-1][0],
                                     self.history_center_points[-1][1],
                                     self.history_heading[-1])
            temp_distance = result[3]
        else:
            temp_distance = self.around_lanes[self.cur_lane_id].width - abs(self.lane_lateral_diff[-1])

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
        # 画出拟合后的曲线
        # print(matAA)
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


        # ax.plot(xxa, yya, color='g', linestyle='-', marker='')
        #
        # ax.legend()
        # plt.show()













"""
feature_values->push_back(theta_filtered);
  feature_values->push_back(theta_mean);
  feature_values->push_back(theta_filtered - theta_mean);
  feature_values->push_back(angle_diff);
  feature_values->push_back(angle_diff_rate);

  feature_values->push_back(lane_l_filtered);
  feature_values->push_back(lane_l_mean);
  feature_values->push_back(lane_l_filtered - lane_l_mean);
  feature_values->push_back(lane_l_diff);
  feature_values->push_back(lane_l_diff_rate);

  feature_values->push_back(speed_mean);
  feature_values->push_back(acc);

  feature_values->push_back(dist_lbs.front());
  feature_values->push_back(dist_lb_rate);
  feature_values->push_back(dist_lb_rate_curr);

  feature_values->push_back(dist_rbs.front());
  feature_values->push_back(dist_rb_rate);
  feature_values->push_back(dist_rb_rate_curr);

  feature_values->push_back(lane_types.front() == 0 ? 1.0 : 0.0);
  feature_values->push_back(lane_types.front() == 1 ? 1.0 : 0.0);
  feature_values->push_back(lane_types.front() == 2 ? 1.0 : 0.0);
  feature_values->push_back(lane_types.front() == 3 ? 1.0 : 0.0);


"""





# # 进行曲线拟合
# matA = []
# for i in range(0, order + 1):
#     matA1 = []
#     for j in range(0, order + 1):
#         tx = 0.0
#         for k in range(0, len(xa)):
#             dx = 1.0
#             for l in range(0, j + i):
#                 dx = dx * xa[k]
#             tx += dx
#         matA1.append(tx)
#     matA.append(matA1)
#
# # print(len(xa))
# # print(matA[0][0])
# matA = np.array(matA)
# # print(matA)
#
# matB = []
# for i in range(0, order + 1):
#     ty = 0.0
#     for k in range(0, len(xa)):
#         dy = 1.0
#         for l in range(0, i):
#             dy = dy * xa[k]
#         ty += ya[k] * dy
#     matB.append(ty)
#
# matB = np.array(matB)
#
# matAA = np.linalg.solve(matA, matB)
#
# print(matAA)


# 从参数FLAGS_prediction_duration(默认5.0s)和FLAGS_prediction_period(默认0.1s)