#!/usr/bin/env python3
from xml.sax.handler import feature_external_ges
import rospy
from geometry_msgs.msg import Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from std_msgs.msg import String


class WaypointFollower(object):

    def __init__(self):
        self.waypoint_pose_sub = rospy.Subscriber("/waypoint_pose", Pose ,self.waypoint_poses_callback, queue_size=1)
        self.waypoint_goal = MoveBaseGoal()
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        self.request_next_activator = rospy.Publisher("/request_next", String , queue_size=1)
        self.request_next_activator.publish("0")
        self.next_goal = MoveBaseGoal()
        self.action = 1
        self.wp_count = 0
        self.check_action()


    def check_action(self):
        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.wp_count == 0:
                self.request_next_waypoint()
            else:
                if self.action == 0:
                    #busy waiting
                    self.request_next_activator.publish("0")
                if self.move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                    self.request_next_waypoint()
                    self.action = 0
            r.sleep()
                


    def waypoint_poses_callback(self, waypoint_pose):
        print("Waypoint received")
        self.wp_count += 1
        self.waypoint_goal.target_pose.pose = waypoint_pose
        self.waypoint_goal.target_pose.header.stamp = rospy.Time.now()
        self.waypoint_goal.target_pose.header.frame_id = 'map'
        print(self.waypoint_goal)
        self.send_next_goal()
        
    

    def request_next_waypoint(self):
        self.request_next_activator.publish("1")
    
    def send_next_goal(self):
        self.move_base_client.send_goal(self.waypoint_goal)
        wait = self.move_base_client.wait_for_result()
        self.action = 0
        if not wait:
            rospy.logerr("Action server not available!")
            #rospy.signal_shutdown("Action server not available!")
        else:
            self.action = self.move_base_client.get_result()


        # for waypoint in range(0, len(waypoint_pose_array_msg.poses)):

        #     print('Waypoint: ' + str(waypoint + 1)
        #     self.waypoint_goal.target_pose.pose = waypoint_pose_array_msg.poses[waypoint]
        #     self.waypoint_goal.target_pose.header.stamp = rospy.Time.now()
        #     self.waypoint_goal.target_pose.header.frame_id = 'map'
        #     self.move_base_client.send_goal(self.waypoint_goal)

            # wait = self.move_base_client.wait_for_result()
            # if not wait:
            #     rospy.logerr("Action server not available!")
            #     rospy.signal_shutdown("Action server not available!")
            # else:
            #     return self.move_base_client.get_result()


if __name__ == '__main__':
    rospy.init_node('waypoint_follower', anonymous=True)
    Result = WaypointFollower()
    # if Result == 1:
    #     print('Waypoint_follower node successfully executed')
    #     Result.request_next_activator.publish("1")
    #     Result = 0
    # else:
    #     Result.request_next_activator.publish("0")
    rospy.spin()
