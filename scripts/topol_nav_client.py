#! /usr/bin/env python

import rospy
import sys
# Brings in the SimpleActionClient
import actionlib
import topological_navigation.msg


def topol_nav_client(orig, targ):
    
    client = actionlib.SimpleActionClient('topological_navigation', topological_navigation.msg.GotoNodeAction)
    
    client.wait_for_server()
    rospy.loginfo(" ... Init done")

    navgoal = topological_navigation.msg.GotoNodeGoal()

    print "Requesting Navigation From %s to %s" %(orig, targ)

    navgoal.target = targ
    navgoal.origin = orig

    # Sends the goal to the action server.
    client.send_goal(navgoal)

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    # Prints out the result of executing the action
    return client.get_result()  # A FibonacciResult

if __name__ == '__main__':
    print 'Argument List:',str(sys.argv)
    if len(sys.argv) < 3 :
	sys.exit(2)
    rospy.init_node('topol_nav_test')
    ps = topol_nav_client(sys.argv[1], sys.argv[2])
    print ps
