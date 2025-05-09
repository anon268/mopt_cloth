import cv2
import rospy
import actionlib
import tf2_ros
import lr_gym
import lr_gym.utils.utils
import lr_gym_utils.msg

import numpy as np

from real_agent import Agent
from cv_bridge import CvBridge
from pyquaternion import Quaternion
from gp_generator import GPGenerator
from learn_assist_pipeline.srv import *

from lr_gym_utils.srv import AddCollisionBox

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from franka_msgs.srv import (
    SetForceTorqueCollisionBehavior,
    SetForceTorqueCollisionBehaviorRequest,
)


REST_POSE = [-0.02, -0.36, 0.04, -2.15, 0.00, 1.82, 0.84]


class GripperWrap(object):
    def __init__(self, synchronous: bool = True):
        self._gripper_action_client = self._connectRosAction(
            "/franka_gripper/gripper_action", GripperCommandAction
        )
        self._waitOnStepCallbacks = []
        self.synchronous = synchronous

    def _connectRosAction(self, actionName: str, msgType):
        ac = actionlib.SimpleActionClient(actionName, msgType)
        rospy.loginfo("Waiting for action " + ac.action_client.ns + "...")
        ac.wait_for_server()
        rospy.loginfo(ac.action_client.ns + " connected.")
        return ac

    def open(self):
        self.controlGripper(0.075, 10)

    def close(self):
        self.controlGripper(0.002, 10)

    def controlGripper(self, width: float, max_effort: float):
        # From Carlo Rizzardo

        # ggLog.info(f"Setting gripper action: width = {width}, max_effort = {max_effort}")
        goal = GripperCommandGoal()
        goal.command.position = width / 2
        goal.command.max_effort = max_effort
        self._gripper_action_client.send_goal(goal)

        def waitCallback():
            r = self._gripper_action_client.wait_for_result(timeout=rospy.Duration(3.0))
            if r:
                if self._gripper_action_client.get_result().reached_goal:
                    return
                else:
                    rospy.logerr(
                        f"Gripper failed to reach goal. Goal={goal}. result = "
                        + str(self._gripper_action_client.get_result())
                    )
            else:
                self._gripper_action_client.cancel_goal()
                self._gripper_action_client.cancel_all_goals()
                r = self._gripper_action_client.wait_for_result(
                    timeout=rospy.Duration(5.0)
                )
                if r:
                    if not self._gripper_action_client.get_result().reached_goal:
                        rospy.logerr(
                            f"Failed to perform gripper move: action timed out. Action canceled.\n Result = {_gripper_action_client.get_result()}\n"
                            + f"goal = {goal}"
                        )
                else:
                    rospy.logerr(
                        "Failed to perform gripper move: action timed out. Action failed to cancel.\n"
                        + f"goal = {goal}"
                    )

        if self.synchronous:
            waitCallback()
        else:
            self._waitOnStepCallbacks.append(waitCallback)

    def completeAllMovements(self) -> int:
        actionFailed = 0
        for callback in self._waitOnStepCallbacks:
            try:
                callback()
            except Exception as e:
                rospy.loginfo(
                    f"Moveit action failed to complete (this is not necessarily a bad thing). step_count ={self._step_count} exception = "
                    + str(e)
                )
                actionFailed += 1
                # time.sleep(5)
        self._waitOnStepCallbacks.clear()
        return actionFailed


def deproj_client(point, depth_image):
    try:
        deproj = rospy.ServiceProxy("deproj_rs", DeprojRs)
        resp = deproj(depth_image, point)
        return resp.point
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def connectRosAction(action_name: str, msgType):
    ac = actionlib.SimpleActionClient(action_name, msgType)
    rospy.loginfo("Waiting for action " + ac.action_client.ns + "...")
    ac.wait_for_server()
    rospy.loginfo(ac.action_client.ns + " connected.")
    return ac


def waitMove(move_ee_client, goal):
    r = move_ee_client.wait_for_result()
    if r:
        if move_ee_client.get_result().succeded:
            # ggLog.info("waited cartesian....")
            return
        else:
            raise RuntimeError(
                f"Failed to move to cartesian pose. Goal={goal}. result = {str(move_ee_client.get_result())}"
            )
    else:
        move_ee_client.cancel_goal()
        move_ee_client.cancel_all_goals()
        r = move_ee_client.wait_for_result(timeout=rospy.Duration(10.0))
        if r:
            raise RuntimeError(
                f"Failed to move to cartesian pose: action timed out. Action canceled. Goal={goal}. Result = {move_ee_client.get_result()}"
            )
        else:
            raise RuntimeError(
                f"Failed to move to cartesian pose: action timed out. Action failed to cancel. Goal={goal}"
            )


def wait_move_retry(move_ee_client, goal, max_retries=5):
    for i in range(max_retries):
        try:
            move_ee_client.send_goal(goal)
            waitMove(move_ee_client, goal)
            return
        except RuntimeError as e:
            rospy.loginfo(
                f"Failed to move to cartesian pose. Retrying... {i+1}/{max_retries}"
            )
            rospy.loginfo(str(e))
    raise RuntimeError(f"Failed to move to cartesian pose. Goal={goal}")


def do_transform(pointStamped, transform):
    outPS = PointStamped()
    outPS.header = pointStamped.header
    qrot = Quaternion(
        transform.transform.rotation.w,
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
    )
    tmat = np.eye(4)
    tmat[:3, :3] = qrot.rotation_matrix
    tmat[:3, 3] = np.array(
        [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ]
    )
    point = np.array(
        [pointStamped.point.x, pointStamped.point.y, pointStamped.point.z, 1.0]
    )
    point = np.matmul(tmat, point)
    outPS.point.x = point[0]
    outPS.point.y = point[1]
    outPS.point.z = point[2]
    return outPS


def setup_panda():
    rospy.wait_for_service("/franka_control/set_force_torque_collision_behavior")
    ftcb_srv = rospy.ServiceProxy(
        "/franka_control/set_force_torque_collision_behavior",
        SetForceTorqueCollisionBehavior,
    )
    ftcb_msg = SetForceTorqueCollisionBehaviorRequest()
    ftcb_msg.lower_torque_thresholds_nominal = [
        20.0,
        20.0,
        18.0,
        18.0,
        16.0,
        14.0,
        12.0,
    ]
    ftcb_msg.upper_torque_thresholds_nominal = [
        20.0,
        20.0,
        18.0,
        18.0,
        16.0,
        14.0,
        12.0,
    ]
    ftcb_msg.lower_force_thresholds_nominal = [
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
    ]  # These will be used by the velocity controller to stop movement
    ftcb_msg.upper_force_thresholds_nominal = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]


rospy.init_node("grasp_routine", anonymous=True)
pub1 = rospy.Publisher("grasp_point", PoseStamped, queue_size=10)
pub2 = rospy.Publisher("grasp_mask", Image, queue_size=10)

agent = Agent()
agent.load("ckpt.pth")
setup_panda()

rospy.loginfo("Adding collision constraints...")
rospy.wait_for_service("/move_helper/add_collision_box")
collision_box = rospy.ServiceProxy("/move_helper/add_collision_box", AddCollisionBox)
c1 = lr_gym.utils.utils.buildPoseStamped([0, -0.75, 0.5], [1, 0, 0, 0], "world")
s1 = Point()
s1.x = 1
s1.y = 1
s1.z = 1
c2 = lr_gym.utils.utils.buildPoseStamped([0, 0, -0.6], [1, 0, 0, 0], "world")
try:
    res = collision_box(c1, s1, False, "", "")
    res = collision_box(c2, s1, False, "", "")
except rospy.ServiceException as e:
    print("Service call failed: %s" % e)
rospy.loginfo("Collision constraints added.")

gripper = GripperWrap()
gripper.open()

rospy.wait_for_service("deproj_rs")
bridge = CvBridge()
tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)

rate = rospy.Rate(10.0)
height = 0.03
grasp_depth = 6e-3

moveEeClient = connectRosAction(
    "/move_helper/move_to_ee_pose", lr_gym_utils.msg.MoveToEePoseAction
)
moveJointClient = connectRosAction(
    "/move_helper/move_to_joint_pose", lr_gym_utils.msg.MoveToJointPoseAction
)

ros_array = lambda x: np.array([x.point.x, x.point.y, x.point.z])
# orientation_xyzw = [0.924, 0.383, 0, 0]
orientation_xyzw = [1, 0, 0, 0]

while not rospy.is_shutdown():
    input("Press enter to start a new grasp")
    try:
        camera_trans = tfBuffer.lookup_transform(
            "world", "camera_color_optical_frame", rospy.Time(0)
        )

        joint_goal = lr_gym_utils.msg.MoveToJointPoseGoal()
        joint_goal.pose = REST_POSE
        joint_goal.velocity_scaling = 0.1
        joint_goal.acceleration_scaling = 0.1
        wait_move_retry(moveJointClient, joint_goal)
        rospy.sleep(1.0)

        goal = lr_gym_utils.msg.MoveToEePoseGoal()
        goal.end_effector_link = "panda_hand_tcp"
        goal.velocity_scaling = 0.1
        goal.acceleration_scaling = 0.1
        goal.do_cartesian = False

        depth_msg = rospy.wait_for_message(
            "/camera/aligned_depth_to_color/image_raw", Image
        )
        color_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        cv_image = bridge.imgmsg_to_cv2(color_msg, desired_encoding="passthrough")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pick, place, qpick, qplace = agent.act(cv_image)
        rospy.loginfo("Agent action collected")

        # Pick
        rospy.loginfo("Picking")
        pt_msg = Point()
        pt_msg.x = pick[0]
        pt_msg.y = pick[1]
        cv_depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        rospy.loginfo(
            f"Image data: {cv_image.shape}, max {cv_image.max()}, min {cv_image.min()}, value at point {cv_image[gp[1], gp[0]]}"
        )
        rospy.loginfo(
            f"Depth data: {cv_depth.shape}, max {cv_depth.max()}, min {cv_depth.min()}, value at point {cv_depth[gp[1], gp[0]]}"
        )

        imgMsg = bridge.cv2_to_imgmsg(qpick, "bgr8")
        pub2.publish(imgMsg)
        rospy.loginfo("Deprojecting...")
        gp3d = deproj_client(pt_msg, depth_msg)
        if gp3d is None:
            rospy.logerr("Failed to deproject grasp point")
            continue
        gp3d = do_transform(gp3d, camera_trans)
        pt3d = np.array([gp3d.point.x, gp3d.point.y, gp3d.point.z + 9e-3 - grasp_depth])

        grasp_pose = lr_gym.utils.utils.buildPoseStamped(
            pt3d, orientation_xyzw, "world"
        )
        pub1.publish(grasp_pose)

        rospy.loginfo(f"Moving over {pt3d}...")
        overdiff = np.array([0.0, 0.0, height])
        goal.pose = lr_gym.utils.utils.buildPoseStamped(
            pt3d + overdiff, orientation_xyzw, "world"
        )
        wait_move_retry(moveEeClient, goal)
        rospy.loginfo(f"Moving to {pt3d}...")
        rospy.sleep(1.0)
        goal.pose = grasp_pose
        goal.do_cartesian = True
        wait_move_retry(moveEeClient, goal)
        rospy.sleep(1.0)
        rospy.loginfo("Closing gripper...")
        gripper.close()
        rospy.sleep(1.0)

        rospy.loginfo(f"Moving up from {pt3d}...")

        goal.pose = lr_gym.utils.utils.buildPoseStamped(
            pt3d + overdiff, orientation_xyzw, "world"
        )
        goal.do_cartesian = True
        wait_move_retry(moveEeClient, goal)
        rospy.sleep(1.0)

        # Place
        rospy.loginfo("Placing")
        pt_msg = Point()
        pt_msg.x = place[0]
        pt_msg.y = place[1]
        cv_depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        rospy.loginfo(
            f"Image data: {cv_image.shape}, max {cv_image.max()}, min {cv_image.min()}, value at point {cv_image[gp[1], gp[0]]}"
        )
        rospy.loginfo(
            f"Depth data: {cv_depth.shape}, max {cv_depth.max()}, min {cv_depth.min()}, value at point {cv_depth[gp[1], gp[0]]}"
        )

        imgMsg = bridge.cv2_to_imgmsg(qplace, "bgr8")
        pub2.publish(imgMsg)
        rospy.loginfo("Deprojecting...")
        gp3d = deproj_client(pt_msg, depth_msg)
        if gp3d is None:
            rospy.logerr("Failed to deproject grasp point")
            continue
        gp3d = do_transform(gp3d, camera_trans)
        pt3d = np.array([gp3d.point.x, gp3d.point.y, 0])

        grasp_pose = lr_gym.utils.utils.buildPoseStamped(
            pt3d, orientation_xyzw, "world"
        )
        pub1.publish(grasp_pose)

        rospy.loginfo(f"Moving over {pt3d}...")
        overdiff = np.array([0.0, 0.0, height])
        goal.pose = lr_gym.utils.utils.buildPoseStamped(
            pt3d + overdiff, orientation_xyzw, "world"
        )
        wait_move_retry(moveEeClient, goal)
        rospy.sleep(1.0)
        gripper.open()

    except Exception as e:
        rospy.logerr(f"Exception: {e}")
        input("Press enter to continue")
    finally:
        gripper.open()
        rate.sleep()
