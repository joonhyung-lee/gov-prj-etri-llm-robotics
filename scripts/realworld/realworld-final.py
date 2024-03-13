#%%
import json
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import mujoco
import mujoco_viewer
import cv2,os
import copy
import sys
sys.path.append('../../')
""" FOR Vision and Planning (Smoothing) """
from models.utils.util_vision import compute_xyz
from models.utils.util import get_interp_const_vel_traj
from models.utils.grpp import GaussianRandomPathClass, kernel_levse
""" FOR MuJoCo """
from models.utils.util import  *
from models.env.manipulator_agent import ManipulatorAgent
""" FOR ONROBOT RG2 """
from pymodbus.client.sync import ModbusTcpClient  
""" FOR MODERN DRIVER """
import roslib; roslib.load_manifest('ur_driver')
import rospy
""" FOR UR5 """
from models.realrobot.gripper import openGrasp, closeGrasp, resetTool
from models.realrobot.RealRobot import RealRobot
from control_msgs.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState

def solve_ik_for_target_position(env, source_obj_position, target_obj_position, q_init):
    # env.init_viewer(viewer_title='UR5e with RG2 gripper and objects', viewer_width=1200, viewer_height=800,
    #                 viewer_hide_menus=True)
    # env.update_viewer(azimuth=124.08, distance=4.00, elevation=-33, lookat=[0.1, 0.05, 0.86],
    #                 VIS_TRANSPARENT=False, VIS_CONTACTPOINT=True,
    #                 contactwidth=0.05, contactheight=0.05, contactrgba=np.array([1, 0, 0, 1]),
    #                 VIS_JOINT=False, jointlength=0.2, jointwidth=0.05, jointrgba=[0.2, 0.6, 0.8, 1.0])
    DO_RENDER = False
    env.reset()  # reset
    env.forward(q=q_init, joint_idxs=env.idxs_forward)
    
    # 0. Solve IK for the init-grasp position
    p_base = env.get_p_body(body_name='ur_base')
    p_trgt = np.array([0.65, 0, 0.9])
    R_trgt = rpy2r(np.radians([-115, 0, 90])) # Up-Right Grasping
    print(f"[INIT GRSAP] Position {p_trgt}")
    q_init_grasp, ik_done = env.solve_ik_repel(
    body_name='ur_tcp_link',p_trgt=p_trgt,R_trgt=R_trgt,
    IK_P=True,IK_R=True, q_init=q_init,idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian,
    RESET=False,DO_RENDER=DO_RENDER,render_every=1,th=1*np.pi/180.0,err_th=1e-3, stepsize = 2.0, repulse=30, BREAK_TICK=2000)

    # 1. Solve IK for the pre-grasping position
    p_base = env.get_p_body(body_name='ur_base')
    p_trgt = source_obj_position.copy()
    p_trgt[0] = 0.70
    p_trgt[2] = 1.05
    print(f"[Pre GRSAP] Position {p_trgt}")
    # R_trgt = rpy2r(np.radians([-180, 0, 90])) # Forward Grasping
    R_trgt = rpy2r(np.radians([-90, 0, 90])) # Up-Right Grasping
    q_pre_grasp, ik_done = env.solve_ik_repel(
    body_name='ur_tcp_link',p_trgt=p_trgt,R_trgt=R_trgt,
    IK_P=True,IK_R=True, q_init=q_init_grasp,idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian,
    RESET=False,DO_RENDER=DO_RENDER,render_every=1,th=1*np.pi/180.0,err_th=1e-3, stepsize = 2.0, repulse=30, BREAK_TICK=2000)
    
    # 2. Solve IK for the grasping position
    p_trgt = source_obj_position.copy()
    if source_obj_position[2] <0.78:
        p_trgt[2] = p_base[2] - 0.02
    elif source_obj_position[2] >= 0.78 and source_obj_position[2] < 0.84:
        p_trgt[2] = p_base[2] + 0.05
    else:
        p_trgt[2] = p_base[2] + 0.074
    print(f"[GRSAP] Position {p_trgt}")
    R_trgt = rpy2r(np.radians([-90, 0, 90])) # Up-Right Grasping
    q_grasp, ik_done = env.solve_ik_repel(
    body_name='ur_tcp_link',p_trgt=p_trgt,R_trgt=R_trgt,
    IK_P=True,IK_R=True, q_init=q_pre_grasp,idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian,
    RESET=False,DO_RENDER=DO_RENDER,render_every=1,th=1*np.pi/180.0,err_th=1e-3, stepsize = 2.0, repulse=30, BREAK_TICK=2000)

    # 3. Solve IK for the lift-up position
    p_trgt = source_obj_position.copy()
    p_trgt[0] = 0.70
    p_trgt[2] = 1.05
    print(f"[LIFT-UP] Position {p_trgt}")
    R_trgt = rpy2r(np.radians([-90, 0, 90])) # Up-Right Grasping
    q_liftup, ik_done = env.solve_ik_repel(
    body_name='ur_tcp_link',p_trgt=p_trgt,R_trgt=R_trgt,
    IK_P=True,IK_R=True, q_init=q_grasp,idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian,
    RESET=False,DO_RENDER=DO_RENDER,render_every=1,th=1*np.pi/180.0,err_th=1e-3, stepsize = 2.0, repulse=30, BREAK_TICK=2000)

    # 4. Solve IK for the place position
    p_trgt = target_obj_position.copy()
    if source_obj_position[2] <0.78:
        p_trgt[2] = p_base[2] - 0.02
    elif source_obj_position[2] >= 0.78 and source_obj_position[2] < 0.84:
        p_trgt[2] = p_base[2] + 0.07
    else:
        p_trgt[2] = p_base[2] + 0.0835
    print(f"[PLACE] Position {p_trgt}")
    R_trgt = rpy2r(np.radians([-90, 0, 90])) # Up-Right Grasping
    q_place, ik_done = env.solve_ik_repel(
    body_name='ur_tcp_link',p_trgt=p_trgt,R_trgt=R_trgt,
    IK_P=True,IK_R=True, q_init=q_liftup,idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian,
    RESET=False,DO_RENDER=DO_RENDER,render_every=1,th=1*np.pi/180.0,err_th=1e-3, stepsize = 2.0, repulse=30, BREAK_TICK=2000)

    # 5. Solve IK for the post-place position
    p_trgt = target_obj_position.copy()
    p_trgt[2] = 0.95
    print(f"[PLACE] Position {p_trgt}")
    R_trgt = rpy2r(np.radians([-90, 0, 90])) # Up-Right Grasping
    q_post_place, ik_done = env.solve_ik_repel(
    body_name='ur_tcp_link',p_trgt=p_trgt,R_trgt=R_trgt,
    IK_P=True,IK_R=True, q_init=q_place,idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian,
    RESET=False,DO_RENDER=DO_RENDER,render_every=1,th=1*np.pi/180.0,err_th=1e-3, stepsize = 2.0, repulse=30, BREAK_TICK=2000)

    # Close viewer
    env.close_viewer()
    print("IK done.")
    print("q_grasp:%s" % (np.degrees(q_grasp)))
    print("q_liftup:%s" % (np.degrees(q_liftup)))
    print("q_place:%s" % (np.degrees(q_place)))
    
    return [q_init_grasp, q_pre_grasp, q_grasp, q_liftup, q_place, q_post_place, q_init]

#%%
"""
    0: Initial Pose
"""
""" Sync with Robot and MuJoCo """
rospy.init_node('Pick_n_Place')
real_robot = RealRobot()
graspclient = ModbusTcpClient('192.168.0.4')
# (-1.1693690458880823, -2.1257835827269496, 2.4137819449054163, 0.9681593614765625, 0.9362161159515381, -1.068995777760641)
real_robot.move_capture_pose(init_q=np.array([-1.1693690458880823, -2.1257835827269496, 2.4137819449054163, 0.9681593614765625, 0.9362161159515381, -1.068995777760641]));time.sleep(1)
resetTool(graspclient)
openGrasp(force=200, width=1000, graspclient=graspclient)
print(real_robot.joint_list.position)
joint_value = real_robot.joint_list.position
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
    1: Capture Init Scene
"""
print("##" * 30)
print("Capture Init Scene")
print("##" * 30)
# Load camera settings
jsonObj = json.load(open("./cam_setting.json"))
stream_width = int(jsonObj["viewer"]['stream-width'])
stream_height = int(jsonObj["viewer"]['stream-height'])
stream_fps = int(jsonObj["viewer"]['stream-fps'])

# Configure pipeline
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, stream_width, stream_height, rs.format.z16, stream_fps)
cfg.enable_stream(rs.stream.color, stream_width, stream_height, rs.format.bgr8, stream_fps)
pipe.start(cfg)
profile = pipe.get_active_profile()
# intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Create the intrinsic matrix
K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
              [0, intrinsics.fy, intrinsics.ppy],
              [0, 0, 1]])
camera_info = [intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy]
print(f"Intrinsic matrix: \n{K}")

# Capture frames
for _ in range(100):
    pipe.wait_for_frames()

# Retrieve frames
frameset = pipe.wait_for_frames()
aligned_frames = rs.align(rs.stream.color).process(frameset)
depth_frame = aligned_frames.get_depth_frame()
color_frame = aligned_frames.get_color_frame()

# Apply depth filters
spatial = rs.spatial_filter()
spatial.set_option(rs.option.holes_fill, 3)
hole_filling = rs.hole_filling_filter()

# Get RGBD images
depth_image = np.asanyarray(hole_filling.process(spatial.process(depth_frame)).get_data())
depth_image_unfilled = np.asanyarray(depth_frame.get_data())
rgb_image = np.asanyarray(color_frame.get_data())
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
# Clip depth image
depth_scale = pipe.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
clipping_distance_in_meters = 2
clipping_distance = clipping_distance_in_meters / depth_scale
depth_unfilled_clipped = np.where((depth_image_unfilled > clipping_distance) | (depth_image_unfilled <= 0), 0, depth_image_unfilled)
depth_clipped = np.where((depth_image > clipping_distance) | (depth_image <= 0), 0, depth_image)
# Cleanup
pipe.stop()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#%%
"""
    2: Click on the object to get the 3D coordinates
"""
print("##" * 30)
print("Click on the object to get the 3D coordinates")
print("##" * 30)

xml_path = '../asset/ur5e/realworld.xml'
MODE = 'window' ################### 'window' or 'offscreen'
env = ManipulatorAgent(rel_xml_path=xml_path,VERBOSE=False, MODE=MODE)
env.close_viewer()

# Move tables and robot base
env.model.body('base_table').pos = np.array([0,0,0.395])
env.model.body('front_object_table').pos = np.array([-38+0.6,-80,0])
env.model.body('side_object_table').pos = np.array([-0.38-2.4,0,0])
env.model.body('side_short_object_table').pos = np.array([0.38+0.4,0,0])

env.model.body('ur_base').pos = np.array([0.18,0,0.79])
for body_name in ['base_table','front_object_table','side_object_table']:
    geomadr = env.model.body(body_name).geomadr[0]
    env.model.geom(geomadr).rgba[3] = 1.0
print ("Ready to Start")
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""" Get 3D coordinates """
xyz_img = compute_xyz(depth_clipped, camera_info=camera_info)
print(f"Shape of xyz_img: {xyz_img.shape}")
rsz_rate = 25
if rsz_rate is not None:
    h_rsz,w_rsz = depth_clipped.shape[0]//rsz_rate,depth_clipped.shape[1]//rsz_rate
    xyz_img_rsz = cv2.resize(xyz_img,(w_rsz,h_rsz),interpolation=cv2.INTER_NEAREST)
env.forward(q=joint_value, joint_idxs=env.idxs_forward)
p_cam, R_cam = env.get_pR_body('ur_camera_center')
p_cam = env.get_p_body("ur_camera_center") + np.array([-0.0, 0.05,0.025])
R_world = env.get_R_body('ur_camera_center')
rotation_mat = np.eye(4)
Transform_rel = rpy2r(np.array([0, -0.5, 0.5])*np.pi)
rotation_mat[:3,:3] =  R_world @ Transform_rel
T_cam = pr2t(p_cam,rotation_mat[:3,:3])
# To world coordinate
xyz_transpose = np.transpose(xyz_img,(2,0,1)).reshape(3,-1) # [3 x N]
xyz_transpose_rsz = np.transpose(xyz_img_rsz,(2,0,1)).reshape(3,-1) # [3 x N]
xyzone_transpose = np.vstack((xyz_transpose,np.ones((1,xyz_transpose.shape[1])))) # [4 x N]
xyzone_transpose_rsz = np.vstack((xyz_transpose_rsz,np.ones((1,xyz_transpose_rsz.shape[1])))) # [4 x N]
xyzone_world_transpose = T_cam @ xyzone_transpose
xyzone_world_transpose_rsz = T_cam @ xyzone_transpose_rsz
xyz_world_transpose = xyzone_world_transpose[:3,:] # [3 x N]
xyz_world_transpose_rsz = xyzone_world_transpose_rsz[:3,:] # [3 x N]
xyz_world = np.transpose(xyz_world_transpose,(1,0)) # [N x 3]
xyz_world_rsz = np.transpose(xyz_world_transpose_rsz,(1,0)) # [N x 3]
xyz_img_world = xyz_world.reshape(xyz_img.shape[0],xyz_img.shape[1],3)
xyz_img_world_rsz = xyz_world_rsz.reshape(xyz_img_rsz.shape[0],xyz_img_rsz.shape[1],3)
print(f"Shape of xyz_img_world: {xyz_img_world.shape}")
print(f"Shape of xyz_img_world_rsz: {xyz_img_world_rsz.shape}")
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""" Click on the object """
clicked_coordinates = []
source_obj_position = None
target_obj_position = None

def mouse_click(event, x, y, flags, param):
    global source_obj_position, target_obj_position
    # Source Object
    if event == cv2.EVENT_LBUTTONDOWN:
        source_obj_position = xyz_img_world[y, x]
        clicked_coordinates.append((x, y))
        print(f"Clicked at: x={x}, y={y}")
        idx = len(clicked_coordinates) - 1
        cv2.circle(rgb_image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(rgb_image, f"{idx}", (x-5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(f"XYZ_WORLD: {xyz_img_world[y, x]}")

    # Target Object
    if event == cv2.EVENT_MBUTTONDOWN:
        target_obj_position = xyz_img_world[y, x]
        clicked_coordinates.append((x, y))
        print(f"Clicked at: x={x}, y={y}")
        idx = len(clicked_coordinates) - 1
        cv2.circle(rgb_image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(rgb_image, f"{idx}", (x-5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(f"XYZ_WORLD: {xyz_img_world[y, x]}")

# Display the RGB image
cv2.namedWindow('Tag Detection Example', cv2.WINDOW_AUTOSIZE)
cv2.imshow("RGB Image", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
cv2.setMouseCallback("RGB Image", mouse_click)

while True:
    cv2.imshow("RGB Image", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
# Cleanup
cv2.destroyAllWindows()
print(f"Source Object Position: {source_obj_position}")
print(f"Target Object Position: {target_obj_position}")
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#%%
"""
    3: Pick-n-Place based on the clicked 3D coordinates.
"""
print("##" * 30)
print("Pick-n-Place based on the clicked 3D coordinates.")
print("##" * 30)
q_init = np.array(joint_value)
q_res = solve_ik_for_target_position(env, source_obj_position.copy(), target_obj_position.copy(), q_init)
q_array = np.array(q_res)
print("planned joint trajectory:",q_array)

""" Smoothing the joint trajectory """
q_traj_list = []
times_list = []
q_array_flatten = np.concatenate([q_init.reshape(1,6) ,q_array.reshape(-1,6)])
for q_before, q_after in zip(q_array_flatten[:-1], q_array_flatten[1:]):
    q_array_ = np.vstack([q_before, q_after])
    times, q_traj = get_interp_const_vel_traj(q_array_, vel=np.radians(15), HZ=env.HZ)
    print("Joint trajectory ready. duration:[%.2f]sec" % (times[-1]))
    q_traj_list.append(q_traj)
    times_list.append(times)
q_traj = np.concatenate(q_traj_list)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """ Excute the joint trajectory in MuJoCo (You can Skip this part) """
# env.init_viewer(viewer_title='UR5e with RG2 gripper and objects',viewer_width=1200,viewer_height=800,
#                 viewer_hide_menus=True)
# env.update_viewer(azimuth=124.08,distance=4.00,elevation=-33,lookat=[0.1,0.05,0.86],
#                   VIS_TRANSPARENT=False,VIS_CONTACTPOINT=False,
#                   contactwidth=0.05,contactheight=0.05,contactrgba=np.array([1,0,0,1]),
#                   VIS_JOINT=False,jointlength=0.2,jointwidth=0.05,jointrgba=[0.2,0.6,0.8,1.0])

# env.reset() # reset
# env.forward(q=joint_value, joint_idxs=env.idxs_forward)
# env.update_viewer(azimuth=170,distance=2.5,lookat=env.get_p_body(body_name='ur_tcp_link'))

# OPEN = True
# CLOSE = False
# grasp_list = [OPEN, OPEN, CLOSE, CLOSE, CLOSE, OPEN]

# for i,q_traj_ in enumerate(q_traj_list):
#     for q_ in q_traj_:
#         if not env.is_viewer_alive():
#             break
#         start_time = env.get_sim_time()
#         env.step(ctrl=q_,ctrl_idxs=env.idxs_step)
#         env.step(ctrl=float(grasp_list[i%6]),ctrl_idxs=6)
#         env.render(render_every=10)
#         if env.loop_every(HZ=1):
#             scene_img = env.grab_image()
#         if env.loop_every(HZ=10):
#             for p in xyz_img_world_rsz.reshape(-1,3):
#                 env.plot_sphere(p,r=0.005,rgba=[0,1,0,1])
# # Close viewer
# env.close_viewer()
# print ("Done.")
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#%%
"""
    4: Pick-n-Place in Real World
"""
print("##" * 30)
print("Pick-n-Place in Real World")
print("##" * 30)

""" Reset to Init capture pose """
q = copy.deepcopy(joint_value)
q_traj = JointTrajectory()
point = JointTrajectoryPoint()
point.positions = q
point.velocities = [0 for _ in range(6)]
point.time_from_start = rospy.Duration.from_sec(1.0)

q_traj.points.append(point)
real_robot.execute_arm_speed(q_traj, speed_limit=1.0)
real_robot.client.wait_for_result()
openGrasp(force=200, width=1000, graspclient=graspclient) # Open the gripper
time.sleep(1)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""" Execute the joint trajectory in Real World """
unit_time_base = 1.0
track_time = 0

q_curr = rospy.wait_for_message("joint_states", JointState).position
env.forward(q=q_curr,joint_idxs=env.idxs_forward)
x_before = env.get_p_body("ur_tcp_link")
q_before = np.array(q_curr)

speed_limit = 0.3  # speed limit of EE velocity
q_speed_limit = 3.0 # speed limit of joint velocity
openGrasp(force=200, width=1000, graspclient=graspclient)
time.sleep(1)

grasp_list = ['open', 'open', 'close', 'close', 'open', 'open']
q_array_flatten = q_array.reshape(-1,6)
for i, qs in enumerate(q_array_flatten):    
    q_traj = JointTrajectory()
    
    env.forward(q=qs,joint_idxs=env.idxs_forward)
    x_curr = env.get_p_body("ur_tcp_link")
    delta_x = np.linalg.norm(x_curr - x_before)
    delta_q = np.linalg.norm(qs - q_before)
    
    unit_time = max(delta_q/q_speed_limit,max(delta_x/(speed_limit), unit_time_base))
    print(f"unit_time: {unit_time}")
    
    track_time = track_time + unit_time
    point = JointTrajectoryPoint()
    point.positions = qs
    point.velocities = [0 for _ in range(6)]
    if i == 0:
        point.time_from_start = rospy.Duration.from_sec(track_time + 1.0)
        q_traj.points.append(point)
        x_before = x_curr
        real_robot.execute_arm_speed(q_traj, speed_limit=1.6)
    elif i == 1 or i == 2 or i==3 or i == 4:
        point.time_from_start = rospy.Duration.from_sec(track_time + 0.5)
        q_traj.points.append(point)
        x_before = x_curr
        real_robot.execute_arm_speed(q_traj, speed_limit=1.6)
    elif i ==5:
        point.time_from_start = rospy.Duration.from_sec(1.5)
        print(f"111track_time: {track_time}")
        q_traj.points.append(point)
        x_before = x_curr
        real_robot.execute_arm_speed(q_traj, speed_limit=2.5)
    else:
        point.time_from_start = rospy.Duration.from_sec(2.5)
        print(f"22track_time: {track_time}")
        q_traj.points.append(point)
        x_before = x_curr
        real_robot.execute_arm_speed(q_traj, speed_limit=2.5)
    real_robot.client.wait_for_result()
    time.sleep(0.5)

    if i==0 or i==1 or i==4 or i==5 or i==6:
        openGrasp(force=200, width=1000, graspclient=graspclient)
    elif i==2: 
        closeGrasp(force=200, width=600, graspclient=graspclient)

# Set to Initial Pose.
real_robot.move_capture_pose(init_q=joint_value);time.sleep(1)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


#%%



