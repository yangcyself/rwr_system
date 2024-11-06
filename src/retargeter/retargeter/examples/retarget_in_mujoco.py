"""
This script is used to launch rokoko tracker and retarget the hand motion to mujoco hand model and visualize in mujoco
"""

from retargeter import RokokoTracker, Retargeter


if __name__=="__main__":
    import threading
    import mujoco
    import mujoco_viewer
    import time
    import numpy as np

    ip = "0.0.0.0"
    port = 14043
    use_coil = True
    tracker = RokokoTracker(ip=ip, port=port, use_coil=use_coil)
    retargeter = Retargeter(
        # urdf_filepath="/home/chenyu/workspace/vqbet_ws/faive_franka_control/faive_viz/urdf/converted.urdf",
        # mjcf_filepath="/home/chenyu/workspace/faive_ros1_ws/src/faive-integration/model/faive_hand_p0/hand.xml",
        mjcf_filepath="/home/chenyu/workspace/faive_ros1_ws/src/faive-integration/model/faive_hand_p4/hand_p4.xml",
        hand_scheme="p4",
    )
    tracker.start()

    # input("Press enter to start, then press enter to stop")
    # keypoint_positions, timestamp = tracker.get_keypoint_positions()
    # from faive_mano_retarget.utils.visulization import plot_mano_hand, draw_frame
    # from faive_mano_retarget.utils.retarget_utils import get_hand_center_and_rotation, normalize_points_to_hand_local
    # import matplotlib.pyplot as plt
    # # fig_ax = plot_mano_hand(keypoint_positions)
    # # hand_center, rot_matrix = get_hand_center_and_rotation(
    # #     thumb_base= keypoint_positions[1], 
    # #     index_base= keypoint_positions[5], 
    # #     middle_base= keypoint_positions[9], 
    # #     ring_base= keypoint_positions[13], 
    # #     pinky_base= keypoint_positions[17]
    # # )
    # # print("hand center", hand_center)   
    # # print("rotation matrix", rot_matrix)
    # # print("rot_matrix det", np.linalg.det(rot_matrix))
    # # print("rot_matrix_self product", np.dot(rot_matrix, rot_matrix.T))
    # # fig_ax = draw_frame(
    # #     origin=hand_center[0],
    # #     x_axis=rot_matrix[:,0],
    # #     y_axis=rot_matrix[:,1],
    # #     z_axis=rot_matrix[:,2], 
    # #     fig_ax = fig_ax
    # # )

    # keypoint_positions,_ = normalize_points_to_hand_local(keypoint_positions)
    # hand_center, rot_matrix = get_hand_center_and_rotation(
    #     thumb_base= keypoint_positions[1], 
    #     index_base= keypoint_positions[5], 
    #     middle_base= keypoint_positions[9], 
    #     ring_base= keypoint_positions[13], 
    #     pinky_base= keypoint_positions[17],
    #     wrist= keypoint_positions[0]
    # )
    # print("hand center", hand_center)
    # print("rotation matrix", rot_matrix)
    # fig_ax = plot_mano_hand(keypoint_positions)
    # plt.show()
    # exit()

    def worker():
    ## mujoco part
        # model = mujoco.MjModel.from_xml_path("/home/chenyu/workspace/faive_ros1_ws/src/faive-integration/model/faive_hand_p0/hand.xml")
        model = mujoco.MjModel.from_xml_path("/home/chenyu/workspace/faive_ros1_ws/src/faive-integration/model/faive_hand_p4/hand_p4.xml")
        # Create the data object for the simulation
        data = mujoco.MjData(model)
        viewer = mujoco_viewer.MujocoViewer(model, data)
        while tracker.keep_running and viewer.is_alive:
            # print("keypoint positions",tracker.get_keypoint_positions())
            keypoint_positions, timestamp = tracker.get_keypoint_positions()
            joint_angles = retargeter.retarget(keypoint_positions)
            print("joint angles",joint_angles)
            # print(joint_angles)
            print("ctrl commands", data.ctrl[:])
            joint_angles_virt = joint_angles.copy()
            joint_angles_virt[1:] *= 0.5 # all joints except the first one are rolling contact joints
            data.ctrl = (joint_angles_virt / 180.0 * np.pi)
            # Step the simulation forward
            mujoco.mj_step(model, data)
            # Update the viewer
            viewer.render()
            # if use_coil:
            #     print("wrist",tracker.get_wrist_pose())
        viewer.close()
    
    t = threading.Thread(target=worker)
    input("Press enter to start, then press enter to stop")
    t.start()
    input("Press enter to stop")
    tracker.stop()
    t.join()


