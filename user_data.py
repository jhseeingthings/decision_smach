#!/usr/bin/env python

import rospy
import smach
import smach_ros
import multiprocessing as mp
import threading as td

new_data = smach.UserData()
data3 = None
data4 = None

class dataStruct:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# define state Foo
class Foo(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['outcome1', 'outcome2'],
                             input_keys=['sm_counter'],
                             output_keys=['sm_counter'])

    def execute(self, user_data):
        rospy.loginfo('Executing state FOO')
        if user_data.sm_counter < 3:
            user_data.sm_counter = user_data.sm_counter + 1
            return 'outcome1'
        else:
            return 'outcome2'


# define state Bar
class Bar(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['outcome1'],
                             input_keys=['data1', 'sm_counter'], output_keys=['data1'])

    def execute(self, user_data):
        rospy.loginfo('Executing state BAR')
        try:
            rospy.loginfo("inside Bar")
            # update the data every time triggered. inside the smach.
            # user_data.update(new_data)
        except Exception as e:
            rospy.loginfo(e)

        for i in range(5):
            update_data(user_data)
            rospy.loginfo(data3.x)
            rospy.loginfo('Counter = %f' % user_data.data1.x)
            rospy.loginfo('Counter = %f' % user_data.data1.y)
            rospy.loginfo('Counter = %f' % user_data.sm_counter)
            rospy.sleep(1)
        return 'outcome1'


def update_data(user_data):
    user_data.data1 = data3


# update the data periodically. outside the smach.
def userdata_update(user_data):
    global new_data
    for i in range(10):
        new_data.data1 = dataStruct(30 + i, 40)
        new_data.data2 = 1 + i
        user_data.update(new_data)
        rospy.loginfo(new_data.data2)
        rospy.loginfo('updating----')
        try:
            rospy.loginfo(data3.x)
        except Exception as e:
            pass
        rospy.sleep(2)

def fun1():
    rospy.loginfo('inside function 1')
    rospy.loginfo(new_data.keys())
    global data3, data4
    for i in range(10):
        data3 = dataStruct(30 + i, 40)
        data4 = 1 + i
        rospy.loginfo('updating data----')
        rospy.sleep(2)

def main():
    rospy.init_node('smach_example_state_machine')

    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['outcome4'])
    # sm = smach.StateMachine(outcomes=['outcome4'], input_keys=['sm_counter', 'data1'])
    # sm = smach.StateMachine(outcomes=['outcome4'], input_keys=['userdata'])

    sm.userdata.sm_counter = 0
    sm.userdata.data1 = dataStruct(10, 20)
    sm.userdata.data3 = dataStruct(100, 200)
    # Open the container
    with sm:
        # Add states to the container
        smach.StateMachine.add('FOO', Foo(),
                               transitions={'outcome1': 'BAR',
                                            'outcome2': 'outcome4'})
        # smach.StateMachine.add('FOO', Foo(),
        #                        transitions={'outcome1': 'BAR',
        #                                     'outcome2': 'outcome4'},
        #                        remapping={'foo_counter_in': 'sm_counter',
        #                                   'foo_counter_out': 'sm_counter'})
        smach.StateMachine.add('BAR', Bar(),
                               transitions={'outcome1': 'FOO'})

    # Create and start the introspection server
    sis = smach_ros.IntrospectionServer('my_smach_introspection_server', sm, '/SM_ROOT')
    sis.start()

    # Execute SMACH plan
    # outcome = sm.execute()

    # pool = mp.Process(target=sm.execute)
    # p2 = mp.Process(target=userdata_update, args=(sm.userdata,))
    # p3 = mp.Process(target=fun1)

    pool = td.Thread(target=sm.execute)
    #p2 = td.Thread(target=userdata_update, args=(sm.userdata,))
    p3 = td.Thread(target=fun1)
    pool.start()
    #p2.start()
    p3.start()

    pool.join()
    #p2.join()
    p3.join()

    # pool = mp.Pool(processes=2)
    # res = pool.apply_async(sm.execute())
    # pool.apply_async(userdata_update(sm.userdata))

    # Wait for ctrl-c to stop the application
    rospy.spin()
    sis.stop()


if __name__ == '__main__':
    main()
