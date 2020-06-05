#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import smach
import smach_ros
from smach_ros import MonitorState, IntrospectionServer

import threading
from multiprocessing.pool import ThreadPool

"""
标准数据类型
在内存中存储的数据可以有多种类型。

例如，一个人的年龄可以用数字来存储，他的名字可以用字符来存储。

Python 定义了一些标准类型，用于存储各种类型的数据。

Python有五个标准的数据类型：

Numbers（数字）
String（字符串）
List（列表）
Tuple（元组）
Dictionary（字典）
"""

'''
FSM:
    Input:
        map:
            1.
            2.
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
            
        Pose:
            1.curX
            2.curY
            3.curV
            
        Traffic Lights/Signs:
            1.
            
            
            
map:
    roadCurve
'''



# class Foo(smach.State):
# #     def __init__(self):
# #         smach.State.__init__(self, outcomes=['outcome1', 'outcome2'],
# #                     input_keys=['foo_input'],
# #                     output_keys=['foo_output'])
# #     """
# #     注意: 通过input_keys从userdata获得的对象被包装为不变性，
# #     因此一个状态不能调用这些对象上的方法。如果需要一个可变的输入对象，
# #     则必须在input_keys和output_keys中指定相同的密钥。
# #     如果你没有传递对象，或者不需要调用方法或修改它们，
# #     那么应该在input_keys和output_keys中使用惟一名称，以避免混淆和潜在的错误。
# #     """
# #     def execute(self, userdata):
# #         # Do something with userdata
# #         if userdata.foo_input == 1:
# #             return 'outcome1'
# #         else:
# #             userdata.foo_output = 3
# #             return 'outcome2'



def parking_spot_choose_decider():
    pass



def lane_choose_decider():
    pass

def lane_borrow_decider():
    pass

def traffic_rules_decider():
    pass

def speed_limit_decider():
    pass

def path_bound_decider():
    pass

def path_priority_decider():
    pass

def re_global_planning_decider():
    pass



#########################################
class EmergencyBrake(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['brakeOn',
                                               'brakeOff'])
    def execute(self, userdata):

        pass

#########################################
class StartupCheck(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['ready'])
    def execute(self, userdata):
        pass

class Startup(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded'])

    def execute(self, userdata):
        pass

#########################################
class InLaneDriving(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['park', 're_global_planning', 'intersection', 'u_turn'])
    def execute(self, userdata):
        pass


#########################################
class ApproachIntersection(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['finished'])

    def execute(self, userdata):
        pass


class CreepToIntersection(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['finished'])

    def execute(self, userdata):
        pass



class PassIntersection(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded'])

    def execute(self, userdata):
        pass



#########################################
class CreepForOpportunity(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['finished'])

    def execute(self, userdata):
        pass



class ExecuteUTurn(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded'])

    def execute(self, userdata):
        pass



#########################################
class ApproachParkingSpot(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['finished'])

    def execute(self, userdata):
        pass


class Park(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded'])

    def execute(self, userdata):
        pass

#########################################
class StopImmediately(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succeeded'])

    def execute(self, userdata):
        pass



#########################################
class StructuredRoad(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['switch'])

    def execute(self, userdata):
        pass



class UnStructuredRoad(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['switch'])

    def execute(self, userdata):
        pass




def main():
    rospy.init_node('decision_smach')

    # Create the top level SMACH state machine
    sm_top = smach.StateMachine(outcomes=[])

    # Open the container
    with sm_top:

        # Create the sub SMACH state machine
        sm_con = smach.Concurrence(outcomes=['outcome5'], default_outcome='outcome5')
        # {'outcome5':{'FOO': 'outcome2','BAR': 'outcome1'}}表示 FOO 和 BAR 输出都要满足条件才会输出 outcome5

        # Open the container
        with sm_con:

            sm_con_scenario = smach.Concurrence(outcomes = ['outcome4'], default_outcome='outcome4')

            with sm_con_scenario:

                sm_con_scenario_main = smach.StateMachine(outcomes = ['outcome3'])
                with sm_con_scenario_main:

                    sm_scenario_startup = smach.StateMachine(outcomes = ['succeeded'])
                    with sm_scenario_startup:
                        smach.StateMachine.add('STARTUP_CHECK', StartupCheck(), transitions = {'ready': 'EXECUTE_STARTUP'})
                        smach.StateMachine.add('EXECUTE_STARTUP', Startup(), transitions = {'succeeded': 'succeeded'})
                    smach.StateMachine.add('STARTUP', sm_scenario_startup, transitions = {'succeeded': 'LANE_FOLLOW'})

                    sm_scenario_lane_follow = smach.StateMachine(outcomes = ['park', 're_global_planning', 'intersection', 'u_turn'])
                    with sm_scenario_lane_follow:
                        smach.StateMachine.add('IN_LANE_DRIVING', InLaneDriving(), transitions = {'park': 'park',
                                                                                                  're_global_planning': 're_global_planning',
                                                                                                  'intersection': 'intersection',
                                                                                                  'u_turn': 'u_turn'})
                    smach.StateMachine.add('LANE_FOLLOW', sm_scenario_lane_follow, transitions = {'park': 'PARK',
                                                                                                  're_global_planning': 'RE_GLOBAL_PLANNING',
                                                                                                  'intersection': 'INTERSECTION',
                                                                                                  'u_turn': 'U_TURN'})

                    sm_scenario_intersection = smach.StateMachine(outcomes = ['succeeded'])
                    with sm_scenario_intersection:
                        smach.StateMachine.add('APPROACH_INTERSECTION', ApproachIntersection(), transitions = {'finished': 'CREEP_TO_INTERSECTION'})
                        smach.StateMachine.add('CREEP_TO_INTERSECTION', CreepToIntersection(), transitions = {'finished': 'PASS_INTERSECTION'})
                        smach.StateMachine.add('PASS_INTERSECTION', PassIntersection(), transitions = {'succeeded': 'succeeded'})
                    smach.StateMachine.add('INTERSECTION', sm_scenario_intersection, transitions = {'succeeded': 'LANE_FOLLOW'})

                    sm_scenario_u_turn = smach.StateMachine(outcomes = ['succeeded'])
                    with sm_scenario_u_turn:
                        smach.StateMachine.add('CREEP_FOR_OPPORTUNITY', CreepForOpportunity(), transitions = {'finished': 'EXECUTE_U_TURN'})
                        smach.StateMachine.add('EXECUTE_U_TURN', ExecuteUTurn(), transitions = {'succeeded': 'succeeded'})
                    smach.StateMachine.add('U_TURN', sm_scenario_u_turn, transitions = {'succeeded': 'LANE_FOLLOW'})

                    sm_scenario_park = smach.StateMachine(outcomes = ['succeeded'])
                    with sm_scenario_park:
                        smach.StateMachine.add('APPROACH_PARKING_SPOT', ApproachParkingSpot(), transitions = {'finished': 'EXECUTE_PARK'})
                        smach.StateMachine.add('EXECUTE_PARK',Park(), transitions = {'succeeded': 'succeeded'})
                    smach.StateMachine.add('PARK', sm_scenario_park, transitions = {'succeeded': 'STARTUP'})

                    sm_scenario_re_global_Planning = smach.StateMachine(outcomes = ['succeeded'])
                    with sm_scenario_re_global_Planning:
                        smach.StateMachine.add('STOP', StopImmediately(), transitions = {'succeeded': 'succeeded'})
                    smach.StateMachine.add('RE_GLOBAL_PLANNING', sm_scenario_re_global_Planning, transitions = {'succeeded': 'LANE_FOLLOW'})

                sm_con_scenario_unstructured = smach.StateMachine(outcomes = ['outcome2'])
                with sm_con_scenario_unstructured:
                    smach.StateMachine.add('STRUCTURED_ROAD', StructuredRoad(), transitions = {'switch': 'UNSTRUCTURED_ROAD'})
                    smach.StateMachine.add('UNSTRUCTURED_ROAD', UnStructuredRoad(), transitions = {'switch': 'STRUCTURED_ROAD'})


                # Add states to the container
                smach.Concurrence.add('SCENARIO_MAIN', sm_con_scenario_main)
                smach.Concurrence.add('SCENARIO_UNSTRUCTURED', sm_con_scenario_unstructured)

            smach.Concurrence.add('SCENARIO_MANAGER', sm_con_scenario)

            smach.Concurrence.add('EMERGENCY_BRAKE', EmergencyBrake())

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


'''
class DecisionScenario:
    def __init__(self):
        rospy.init_node('decision_smach_scenario')

        # the top level of the state machine
        self.sm_top = smach.StateMachine()

        # define user data in use
        self.sm_top.userdata.road_current = 0
        self.sm_top.userdata.obstacle_selected = 0
        self.sm_top.userdata.pose_current = 0
        self.sm_top.userdata.scenario = 0
        self.sm_top.userdata.behavior = 0
        self.sm_top.userdata.sub_behavior = 0
        self.sm_top.userdata.speed_upper_limit = 0
        self.sm_top.userdata.speed_lower_limit = 0
        self.sm_top.userdata.target_lane_id = 0
        self.sm_top.userdata.re_global_planning = 0

        rospy.init_node('decision_smach')

        # Create the top level SMACH state machine
        sm_top = smach.StateMachine(outcomes=['outcome6'])

        # Open the container
        with sm_top:
            # Create the sub SMACH state machine
            sm_con = smach.Concurrence(outcomes=['outcome4', 'outcome5'],
                                       default_outcome='outcome4',
                                       outcome_map={'outcome5':
                                                        {'FOO': 'outcome2',
                                                         'BAR': 'outcome1'}})
            # {'outcome5':{'FOO': 'outcome2','BAR': 'outcome1'}}表示 FOO 和 BAR 输出都要满足条件才会输出 outcome5

            # Open the container
            with sm_con:
                sm_con_scenario = smach.StateMachine()

                with sm_con_scenario:
                    sm_con_scenario_main = smach.StateMachine()
                    with sm_con_scenario_main:
                        sm_scenario_startup = smach.StateMachine()
                        with sm_scenario_startup:
                            smach.StateMachine.add('STARTUP_CHECK', StartupCheck())
                            smach.StateMachine.add('STARTUP_CHECK', Startup())
                        smach.StateMachine.add('STARTUP', sm_scenario_startup)

                        sm_scenario_lane_follow = smach.StateMachine()
                        with sm_scenario_lane_follow:
                            smach.StateMachine.add('IN_LANE_DRIVING', InLaneDriving())
                        smach.StateMachine.add('LANE_FOLLOW', sm_scenario_lane_follow)

                        sm_scenario_intersection = smach.StateMachine()
                        with sm_scenario_intersection:
                            smach.StateMachine.add('APPROACH_INTERSECTION', ApproachIntersection())
                            smach.StateMachine.add('CREEP_TO_INTERSECTION', CreepToIntersection())
                            smach.StateMachine.add('PASS_INTERSECTION', PassIntersection())
                        smach.StateMachine.add('INTERSECTION', sm_scenario_intersection)

                        sm_scenario_u_turn = smach.StateMachine()
                        with sm_scenario_u_turn:
                            smach.StateMachine.add('CREEP_FOR_OPPORTUNITY', CreepForOpportunity())
                            smach.StateMachine.add('EXECUTE_U_TURN', ExecuteUTurn())
                        smach.StateMachine.add('U_TURN', sm_scenario_u_turn)

                        sm_scenario_park = smach.StateMachine()
                        with sm_scenario_park:
                            smach.StateMachine.add('APPROACH_PARKING_SPOT', ApproachParkingSpot())
                            smach.StateMachine.add('EXECUTE_PARK', Park())
                        smach.StateMachine.add('PARK', sm_scenario_park)

                        sm_scenario_re_global_Planning = smach.StateMachine()
                        with sm_scenario_re_global_Planning:
                            smach.StateMachine.add('STOP', StopImmediately())
                        smach.StateMachine.add('RE_GLOBAL_PLANNING', sm_scenario_re_global_Planning)

                    sm_con_scenario_unstructured = smach.StateMachine()
                    with sm_con_scenario_unstructured:
                        smach.StateMachine.add('STRUCTURED_ROAD', StructuredRoad())
                        smach.StateMachine.add('UNSTRUCTURED_ROAD', UnStructuredRoad())

                    # Add states to the container
                    smach.Concurrence.add('SCENARIO_MAIN', sm_con_scenario_main)
                    smach.Concurrence.add('SCENARIO_UNSTRUCTURED', sm_con_scenario_unstructured)

                smach.Concurrence.add('SCENARIO_MANAGER', sm_con_scenario,
                                      transitions={'outcome4': 'CON',
                                                   'outcome5': 'outcome6'})

                smach.Concurrence.add('EMERGENCY_BRAKE', EmergencyBrake())

            smach.StateMachine.add('FINITE_STATE_MACHINE', sm_con)

        # Create and start the introspection server
        sis = smach_ros.IntrospectionServer('server_name', self.sm_top, '/SM_ROOT')
        sis.start()

        # Create a thread to execute the smach container
        pool = ThreadPool(processes=1)
        async_result = pool.apply_async(self.sm_top.execute())

        # # Execute SMACH plan
        # outcome = self.sm_top.execute()
        rospy.spin()
        sis.stop()
'''
