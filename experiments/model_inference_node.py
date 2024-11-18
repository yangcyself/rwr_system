from time import sleep
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import TransformStamped, PoseStamped
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from threading import Lock

import torch
import yaml
from rwr_system.src.common.utils import numpy_to_float32_multiarray, float32_multiarray_to_numpy
from srl_il.export.il_policy import get_policy_from_ckpt

class CameraListener(Node):
    def __init__(self, camera_topic, name, node):
        self.camera_topic = camera_topic
        self.lock = Lock()
        self.image = None
        self.name = name
        self.im_subscriber = node.create_subscription(
            Image, self.camera_topic, self.recv_im, 10
        )

    def recv_im(self, msg: Image):
        with self.lock:
            self.image = msg

    def get_im(self):
        with self.lock:
            return deepcopy(self.image)

class PolicyPlayerAgent(Node):
    def __init__(self):
        super().__init__("policy_publisher")
        
        self.declare_parameter("camera_topics", [])
        self.declare_parameter("policy_ckpt_path", "")   # assume the policy ckpt is saved with its config
        self.camera_topics = self.get_parameter("camera_topics").value # tuples (topics, names)
        self.policy_ckpt_path = self.get_parameter("policy_ckpt_path").value
        
        self.lock = Lock()

        self.hand_pub = self.create_publisher(
            Float32MultiArray, "/hand/policy_output", 10
        )
        self.hand_sub = self.create_subscription(
            Float32MultiArray, "/hand/policy_output", self.hand_callback, 10
        )
        
        self.arm_publisher = self.create_publisher(
            PoseStamped, "/franka/end_effector_pose_cmd", 10
        )
        self.arm_subscriber = self.create_subscription(
            PoseStamped, "/franka/end_effector_pose", self.arm_pose_callback, 10
        )

        self.camera_listeners = [
            CameraListener(camera_topic, camera_name, self) for camera_topic, camera_name in self.camera_topics
        ]
        
        self.bridge = CvBridge()


        self.policy = get_policy_from_ckpt(self.policy_ckpt_path)
        self.policy.reset_policy()
        self.policy_run = self.create_timer(0.05, self.run_policy_cb) # 20hz
        
        

    def publish(self, hand_policy: np.ndarray, wrist_policy: np.ndarray):
        # publish hand policy
        hand_msg = numpy_to_float32_multiarray(hand_policy)
        self.hand_pub.publish(hand_msg)

        # publish wrist policy
        wrist_msg = PoseStamped()
        wrist_msg.pose.position.x, wrist_msg.pose.position.y, wrist_msg.pose.position.z = wrist_policy[:3]
        wrist_policy_quat = R.from_euler(
            "xyz", wrist_policy[-3:], degrees=False
        ).as_quat()
        (   wrist_msg.pose.orientation.x,
            wrist_msg.pose.orientation.y,
            wrist_msg.pose.orientation.z,
            wrist_msg.pose.orientation.w,
        ) = wrist_policy_quat
        wrist_msg.header.stamp = self.get_clock().now().to_msg()   
        wrist_msg.header.frame_id = "world" 
        self.arm_publisher.publish(wrist_msg)
    
    def arm_pose_callback(self, msg: PoseStamped):
        current_wrist_state_msg = msg.pose
        position = [current_wrist_state_msg.position.x, current_wrist_state_msg.position.y, current_wrist_state_msg.position.z]
        quaternion = [current_wrist_state_msg.orientation.x, current_wrist_state_msg.orientation.y, current_wrist_state_msg.orientation.z, current_wrist_state_msg.orientation.w]
        rotation = R.from_quat(quaternion).as_euler("xyz", degrees=False)
        self.current_wrist_state = np.concatenate([position, rotation])
    
    def hand_callback(self, msg: Float32MultiArray):
        self.current_hand_state = float32_multiarray_to_numpy(msg)

    def get_current_observations(self):
        obs_dict = {}
        get_data_success = True
        
        images = {camera.name: camera.get_im() for camera in self.camera_listeners}
        if any([im is None for im in images.values()]):
            get_data_success = False
            return get_data_success, obs_dict

        images = {
            k: self.bridge.imgmsg_to_cv2(v, "bgr8").transpose(2, 0, 1)/255.0
            for k, v in images.items()
        }

        with self.lock:
            wrist_pose = self.current_wrist_state
            hand_state = self.current_hand_state
        
        qpos = np.concatenate(
            [wrist_pose, hand_state.flatten()]
        )

        obs_dict.update(images)
        obs_dict["qpos"] = qpos
        return get_data_success, obs_dict
    
    def run_policy_cb(self):

        get_data_success, obs_dict = self.get_current_observations()
        if not get_data_success:
            self.get_logger().info("No observations available. Sleeping for 1 seconds.")
            sleep(1)
        with torch.inference_mode():
            obs_dict = {k: torch.tensor(v).float().unsqueeze(0) for k, v in obs_dict.items()} # add batch dimension
            action = self.policy.predict_action(obs_dict)
            wrist_action = action[0, :7].cpu().numpy()
            hand_action = action[0, 7:].cpu().numpy()
        self.publish(wrist_action, hand_action)

def main(args=None):
    rclpy.init(args=args)
    node = PolicyPlayerAgent()
    rclpy.spin(node)
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()
