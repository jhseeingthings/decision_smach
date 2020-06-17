# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from local_messages.msg import Road
from local_messages.msg import GlobalPose
from local_messages.msg import Obstacles
import numpy as np
import math


road_data = None
global_pose_data = None
obstacles_data = None
cur_lane_list = []
UNCLASSIFIED = False
NOISE = None

WIDTH = 1.86
LENGTH = 4.99
LENGTH_REAR = 1.50



def lane_projection(map_x, map_y, map_num, cur_x, cur_y, cur_yaw, type = 0):
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
    global road_data
    road_data = LaneInfo(road_msg)
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def global_pose_callback(global_pose_msg):
    global global_pose_data
    global_pose_data = global_pose_msg

def obstacles_callback(obstacles_msg):
    global obstacles_data
    for i in range(len(obstacles_msg.obstacles)):
        obstacle_projection_result = ObstacleSelection(obstacles_data.obstacles[i])
        obstacles_data.append(obstacle_projection_result)

def listener():
    # 注意node的名字得独一无二，但是topic的名字得和你想接收的信息的topic一样！
    rospy.init_node('listener', anonymous=True)
    # Subscriber函数第一个参数是topic的名称，第二个参数是接受的数据类型 第三个参数是回调函数的名称
    rospy.Subscriber("global_pose", GlobalPose, global_pose_callback)
    rospy.Subscriber("map_road", Road, road_callback)
    rospy.Subscriber("fused_obstacles", Obstacles, obstacles_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()



class LaneInfo:
    def __init__(self, road_msg):
        self.lanes = road_msg.lanes
        self.preferred = []
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
        self.next_stop_x = 0
        self.next_stop_y = 0
        self.next_stop_type = 0
        self.turn_type = 0
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


    def find_current_lane(self):
        for i in range(len(self.lanes)):
            self.points_x = []
            self.points_y = []
            self.preferred = []
            for j in range(len(self.lanes[i].points)):
                self.points_x.append(self.lanes[i].points[j].x)
                self.points_y.append(self.lanes[i].points[j].y)
            self.points_num = len(self.points_x)
            result = lane_projection(self.points_x, self.points_y, self.points_num, global_pose_data.mapX, global_pose_data.mapY, global_pose_data.mapHeading)
            self.preferred.append(self.lanes[i].preferred)
            self.offset.append(result[3])
            self.dir_diff.append(result[4])
            self.before_length.append(result[5])
            self.after_length.append(result[6])

        # choose current lane id and index
        DIR_THRESHOLD = 120.0 / 180.0 * 3.14159265
        OFFSET_THRESHOLD = 1.0
        min_offset = 50
        cur_lane_index = -1
        count = 0
        preferred_index_set = []
        preferred_id_set = []



        # 选择当前车道(找全局规划)
        for i in range(len(self.lanes)):
            abs_offset = abs(self.offset[i])
            abs_dir_diff = abs(self.dir_diff[i])
            # 附近车道，距离最小，方向偏差小，优先级高（2），不是车道末段
            if self.lanes[i].relation == 1 and abs_offset < OFFSET_THRESHOLD \
                    and abs_dir_diff < DIR_THRESHOLD and self.lanes[i].preferred == 2 \
                    and self.after_length[i] > 5:
                count += 1
                preferred_index_set.append(i)
                preferred_id_set.append(self.lanes[i].id)
        if count == 0:
            for i in range(len(self.lanes)):
                abs_offset = abs(self.offset[i])
                abs_dir_diff = abs(self.dir_diff[i])
                # 附近车道，距离最小，方向偏差小，优先级为 1，不是车道末段
                if self.lanes[i].relation == 1 and abs_offset < OFFSET_THRESHOLD \
                        and abs_dir_diff < DIR_THRESHOLD and self.lanes[i].preferred == 1 \
                        and self.after_length[i] > 5:
                    count += 1
                    preferred_index_set.append(i)
                    preferred_id_set.append(self.lanes[i].id)
        if count == 0:
            for i in range(len(self.lanes)):
                abs_offset = abs(self.offset[i])
                abs_dir_diff = abs(self.dir_diff[i])
                #
                if self.lanes[i].relation == 1 and abs_offset < min_offset \
                        and abs_dir_diff < DIR_THRESHOLD and self.lanes[i].preferred > 0 \
                        and self.after_length[i] > 5:
                    count = 1
                    min_offset = abs_offset
                    preferred_index_set[0] = i  # 只有一条
                    preferred_id_set[0] = self.lanes[i].id

        # 对于多条符合要求的 lanes ，选一条(id 最小的一条，来保证连续性）
        if count != 0:
            min_id = 10000
            min_id_index = -1
            for i in range(count):
                if preferred_id_set[i] < min_id:
                    min_id = preferred_id_set[i]
                    min_id_index = preferred_index_set[i]
            cur_lane_index = min_id_index
        if cur_lane_index == -1:
            cur_lane_index = 0

        temp_lead_to_id = -1
        lead_to_index = 0
        for i in range(len(self.lanes[cur_lane_index].leadToIds)):
            for j in range(len(self.lanes)):
                if self.lanes[j].id == self.lanes[cur_lane_index].leadToIds[i] \
                    and self.lanes[j].preferred == 2:
                    temp_lead_to_id = self.lanes[j].id
                    lead_to_index = j
                    cur_joint_index = i
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
                        cur_joint_index = i
                        break
                if temp_lead_to_id != -1:
                    break

        # generate and merge current lane
        for i in range(len(self.lanes[cur_lane_index].points)):
            self.cur_lane_x.append(self.lanes[cur_lane_index].points[i].x)
            self.cur_lane_y.append(self.lanes[cur_lane_index].points[i].y)
        # merge !!!
        if self.after_length < 10:
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


        if self.lanes[cur_lane_index].stopType == 0:
            self.next_stop_x = self.lanes[lead_to_index].nextStop.x
            self.next_stop_y = self.lanes[lead_to_index].nextStop.y
            self.next_stop_type = self.lanes[lead_to_index].stopType
        else:
            self.next_stop_x = self.lanes[cur_lane_index].nextStop.x
            self.next_stop_y = self.lanes[cur_lane_index].nextStop.y
            self.next_stop_type = self.lanes[cur_lane_index].stopType

        self.turn_type = self.lanes[cur_lane_index].turn
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


        cur_lane_list.append(self.cur_lane_id)




class ObstacleSelection:
    def __init__(self, obstacles_perception):
        self.id = obstacles_perception.id
        self.polygon = obstacles_perception.polygon
        self.direction = obstacles_perception.direction
        self.center = obstacles_perception.center # center[0] x, center[1] y
        self.size = obstacles_perception.size # size[0] length , size[1] width
        self.type = obstacles_perception.type
        self.confidence = obstacles_perception.confidence
        self.velocity = obstacles_perception.velocity # velocity[0] x, velocity[1] y, velocity[2] z

        self.heading_angle = 0
        self.s_begin = 0
        self.s_end = 0
        self.l_begin = 0
        self.l_end = 0
        self.s_velocity = 0
        self.l_velocity = 0
        self.obstacle_status = 0
        self.sub_decision = 0
        self.safe_distance = 0
        self.obstacles_projection()

    def obstacles_projection(self):
        s_range = []
        l_range = []
        self.heading_angle =
        self.polygon = np.array([self.center[0]])
        for i in range(len(self.polygon)):
            result = lane_projection(road_data.cur_lane_x, road_data.cur_lane_y, road_data.cur_lane_num, self.polygon[i].x, self.polygon[i].y, self.direction)
            # L: result[3], S: result[5]
            s_range.append(result[5])
            l_range.append(result[3])
        self.s_begin = min(s_range)
        self.s_end = max(s_range)
        self.l_begin = min(l_range)
        self.l_end = max(l_range)
        if self.velocity > 0:
            self.obstacle_status = 1
            result_center = lane_projection(road_data.cur_lane_x, road_data.cur_lane_y, road_data.cur_lane_num, self.center.x, self.center.y, self.direction)
            self.s_velocity = math.cos(result_center[4]) * self.velocity
            self.l_velocity = math.sin(result_center[4]) * self.velocity



# int obstacle_lane_distance(local_messages::Obstacle obstacle, local_messages::Lane lane, local_messages::GlobalPose pose, float &distance , vector<float> &obswidth,int &outputflag);

if __name__ == '__main__':
    listener()
    vehicle_projection = lane_projection(road_data.cur_lane_x, road_data.cur_lane_y, road_data.cur_lane_num, global_pose_data.mapX, global_pose_data.mapY, global_pose_data.mapHeading)
    vehicle_bound = np.array([global_pose_data.mapX, global_pose_data.mapY])

    lane_test = []
    boundaryPoints = []

    for i in range(len(boundaryPoints)):
        result = lane_projection(road_data.cur_lane_x, road_data.cur_lane_y, road_data.cur_lane_num, )








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