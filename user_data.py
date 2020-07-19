#!/usr/bin/env python

import rospy
import smach
import smach_ros
import multiprocessing as mp


class dataStruct:
	def __init__(self, x, y):
		self.x = x
		self.y = y



# define state Foo
class Foo(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['outcome1','outcome2'],
                             input_keys=['foo_counter_in'],
                             output_keys=['foo_counter_out'])

    def execute(self, user_data):
        rospy.loginfo('Executing state FOO')
        if user_data.foo_counter_in < 3:
            user_data.foo_counter_out = user_data.foo_counter_in + 1
            return 'outcome1'
        else:
            return 'outcome2'


# define state Bar
class Bar(smach.State):
    def __init__(self):
        smach.State.__init__(self, 
                             outcomes=['outcome1'],
                             input_keys=['bar_counter_in','bar_counter_in1'])
        
    def execute(self, user_data):
        for i in range(10):
            rospy.loginfo('Executing state BAR')
            rospy.loginfo('Counter = %f'%user_data.bar_counter_in.x)
            rospy.loginfo('Counter = %f'%user_data.bar_counter_in.y)
            rospy.loginfo('Counter = %f'%user_data.bar_counter_in1)
            rospy.sleep(1)
        return 'outcome1'
        

def userdata_update(user_data):
    new_data = smach.UserData()
    for i in range(100):
        new_data.data1 = dataStruct(30+i, 40)
        new_data.data2 = 1+i
        user_data.update(new_data)
        rospy.loginfo('updating----')
        rospy.sleep(2)

def main():
    rospy.init_node('smach_example_state_machine')

    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['outcome4'])
    sm.userdata.sm_counter = 0
    sm.userdata.data1 = dataStruct(10,20)
    # Open the container
    with sm:
        # Add states to the container
        smach.StateMachine.add('FOO', Foo(), 
                               transitions={'outcome1':'BAR', 
                                            'outcome2':'outcome4'},
                               remapping={'foo_counter_in':'sm_counter', 
                                          'foo_counter_out':'sm_counter'})
        smach.StateMachine.add('BAR', Bar(), 
                               transitions={'outcome1':'FOO'},
                               remapping={'bar_counter_in':'data1',
					                      'bar_counter_in1':'sm_counter'})

    # Create and start the introspection server
    sis = smach_ros.IntrospectionServer('my_smach_introspection_server', sm, '/SM_ROOT')
    sis.start()

    # Execute SMACH plan
    # outcome = sm.execute()

    pool = mp.Process(target=sm.execute)
    p2 = mp.Process(target=userdata_update(sm.userdata))
    pool.start()
    p2.start()

    pool.join()
    p2.join()

    # pool = mp.Pool(processes=2)
    # res = pool.apply_async(sm.execute())
    # pool.apply_async(userdata_update(sm.userdata))



    # Wait for ctrl-c to stop the application
    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()
