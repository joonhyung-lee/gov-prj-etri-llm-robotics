#%%
import json
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
import os
import copy
import sys
import cv2
import openai
import argparse
import sys
sys.path.append('../../')
from models.utils.util import  *
from models.utils.util_preference import get_stacked_order, preference_criteria#, evaluate_preference
from models.utils.util_visualize import show_images #, find_closest_objects_3d, plot_closest_pairs_with_labels, get_closest_pairs_img
from models.env.manipulator_agent import ManipulatorAgent
from models.utils.gpt_helper import (GPT4VisionClass,GPTchatClass,set_openai_api_key_from_txt,printmd, response_to_json, parse_and_get_action,
                                     extract_arguments, match_objects, decode_image, parse_actions_to_executable_strings)
print ("openai version:[%s]"%(openai.__version__))
print ("mujoco version:[%s]"%(mujoco.__version__))

#%%
# Parse command line arguments
parser = argparse.ArgumentParser(description='Handle multiple interactions to extract user preferences.')
parser.add_argument('--num_interaction', type=int, default=5, help='Number of interactions')
parser.add_argument('--MODE', type=str, choices=['window', 'offscreen'], default='offscreen', help='Select mode for window or offscreen')
parser.add_argument('--save_dir', type=str, default='./result/user-feedback/', help='Directory to save the interaction history')
parser.add_argument('--model', type=str, default='gpt-4-vision-preview', choices=['gpt-4-vision-preview', 'gpt-4', 'gpt-3.5-turbo'], help='Select GPT model')
args = parser.parse_args(args=[])

#%%
if args.model == 'gpt-4-vision-preview':
    # model = GPT4VisionClass(key_path='../../key/my_key.txt', max_tokens=1024, temperature=0.95,
    model = GPT4VisionClass(key_path='../../key/my_key.txt', max_tokens=1024, temperature=0.95,
                            gpt_model="gpt-4-vision-preview", role_msg="You are a helpful agent with vision capabilities; do not respond to objects not depicted in images."
    )
elif args.model == 'gpt-4':
    set_openai_api_key_from_txt(key_path='../../key/my_key.txt')
    model = GPTchatClass(gpt_model='gpt-4',
                         role_msg='Your are a helpful assistant summarizing infromation and answering user queries.')
elif args.model == 'gpt-3.5-turbo':
    set_openai_api_key_from_txt(key_path='../../key/my_key.txt')
    model = GPTchatClass(gpt_model='gpt-3.5-turbo',
                         role_msg='Your are a helpful assistant summarizing infromation and answering user queries.')

#%%
# 0. Start with MuJoCo simulation
min_distance = 0.15
object_names = ["obj_box_red_01", "obj_box_blue_01", "obj_box_green_01", 
                "obj_bowl_red", "obj_bowl_blue", "obj_bowl_green"]
preference_history = []
selected_user_option_history = []

for interaction_idx in range(args.num_interaction):
    annotator_info = {}
    interaction_info = {}
    len_options = []
    selected_option_idx = []
    history_actions = []
    rgb_img_list = []
    scene_img_list = []
    if interaction_idx == 0:
        print("\033[94m [Start with MuJoCo simulation] \033[00m")
        xml_path = '../../asset/ur5e/scene-w-ground-fb.xml'
        MODE = args.MODE ################### Select mode for 'window' or 'offscreen'
        TOP_VIEW = True
        env = ManipulatorAgent(rel_xml_path=xml_path,VERBOSE=False, MODE=MODE)

        # Joint indices
        rev_joint_names = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
                        'wrist_1_joint','wrist_2_joint','wrist_3_joint']
        idxs_ur_fwd = env.get_idxs_fwd(joint_names=rev_joint_names)
        idxs_ur_jac = env.get_idxs_jac(joint_names=rev_joint_names)
        idxs_ur_step = env.get_idxs_step(joint_names=rev_joint_names)
    else:
        env.DONE = False # Reset the done flag

    # Place objects
    obj_box_names = [body_name for body_name in env.body_names
                if body_name is not None and (body_name.startswith("obj_box"))]
    obj_bowl_names = [body_name for body_name in env.body_names
                if body_name is not None and (body_name.startswith("obj_bowl"))]
    n_obj = 3 # len(obj_names)
    xyzs = sample_xyzs(
        n_sample=n_obj*2,x_range=[0.525,1.15],y_range=[-0.38,0.38],z_range=[0.74,0.85],min_dist=0.2,xy_margin=0.1)

    for obj_idx, obj_box_name, obj_bowl_name in zip(range(n_obj), obj_box_names, obj_bowl_names):
        color_idx = obj_idx % 3
        box_jntadr = env.model.body(obj_box_name).jntadr[0]
        env.model.joint(box_jntadr).qpos0[:3] = xyzs[obj_idx, :]
        bowl_jntadr = env.model.body(obj_bowl_name).jntadr[0]
        env.model.joint(bowl_jntadr).qpos0[:3] = xyzs[obj_idx+n_obj, :]

    # Move tables and robot base
    env.model.body('base_table').pos = np.array([0,0,0.395])
    env.model.body('front_object_table').pos = np.array([-38+0.6,0,0])
    env.model.body('side_object_table').pos = np.array([0.38+0.4,0,0])
    # env.model.body('side_object_table').pos = np.array([-0.05,-0.80,0])
    env.model.body('ur_base').pos = np.array([0.18,0,0.79])
    env.model.body('ur_base').pos = np.array([0.18,0,0.8]) # robot base
    for body_name in ['base_table','front_object_table','side_object_table']:
        geomadr = env.model.body(body_name).geomadr[0]
        env.model.geom(geomadr).rgba[3] = 1.0

    print ("Ready.")

    env.set_viewer(VERBOSE=False)
    env.reset()
    # init_pose = np.array([np.deg2rad(-90), np.deg2rad(-132.46), np.deg2rad(122.85), np.deg2rad(99.65), np.deg2rad(45), np.deg2rad(-90.02)])
    init_pose = np.array([-1.571, -2.1916,  1.7089,  2.0533,  1.1000, -1.5706])
    env.forward(q=init_pose,joint_idxs=idxs_ur_fwd)

    FIRST_FLAG = True
    while env.is_viewer_alive():
        # Step
        env.step(ctrl=np.append(init_pose,1.0),ctrl_idxs=idxs_ur_step+[6])
        
        # Render
        if env.loop_every(HZ=20) or FIRST_FLAG:
            scene_img,rgb_img,depth_img = env.get_image_both_mode(body_name='ur_camera_center')
            env.render(render_every=1)
        
        if env.get_sim_time() >= 5.0:
            # Plot world view and egocentric RGB and depth images
            closest_obj_pairs_img_before, closest_obj_pairs_names_before, closest_obj_pairs_indices_before = env.get_closest_pairs(object_names, TOP_VIEW=TOP_VIEW, min_distance=min_distance)
            if env.MODE == 'offscreen':
                show_images(scene_img, rgb_img, closest_obj_pairs_img_before, None, None, None, title="Initial scene and RGB images")
            break
        # Clear flag
        FIRST_FLAG = False
    env.close_viewer()
    print("[Done] Capture MuJoCo simulation images")
    rgb_img_init = rgb_img.copy()
    scene_img_init = scene_img.copy()
    closest_obj_pairs_img_init = closest_obj_pairs_img_before.copy()
    rgb_img_list.append(rgb_img)
    scene_img_list.append(scene_img)

    # 1. Start with Image description
    # print("\033[94m [Starts with Image description] \033[00m")
    # 2. System prompt: functions, object_names
    if interaction_idx == 0:
        print("\033[94m [System prompt: inform functions and object names] \033[00m")
        model = GPT4VisionClass(key_path='../../key/my_key.txt', max_tokens=1024, temperature=0.95,
        # model = GPT4VisionClass(key_path='../../key/my_key.txt', max_tokens=1024, temperature=0.95,
            gpt_model="gpt-4-vision-preview",
            role_msg="You are a helpful agent with vision capabilities; do not respond to objects not depicted in images."
            )
        # RESET_CHAT = True

        with open('./prompt/common_prompt.txt', 'r') as file:
            common_text = file.read()
        model.set_common_prompt(common_text)
        model._backup_chat()

        query_text = f"I want to rearrange the objects on the table. The object names are {object_names}. Can you give me some options to choose from using only the object name I gave you?"
        response = model.chat(query_text=query_text, image_paths=None, images=[rgb_img],
                                    PRINT_USER_MSG=True,
                                    PRINT_GPT_OUTPUT=True,
                                    RESET_CHAT=False,
                                    RETURN_RESPONSE=True,
                                    DETAIL='high')
    else:
        RESET_CHAT = False
        model.messages = model.init_messages
        common_text = f"""
        The interaction has done. I will give you the previous user's selected options and preference summary: <selected_option_list>: {preference_history}, <preference_summary>: {selected_user_option_history}.
        Generate options concentrate on the given informations. And if you know user's preferred objective, you can perform multiple actions in one option. The response format is same as before
        """
        model.set_common_prompt(common_text)

        query_text = f"In the new image, I want to rearrange the objects on the table. The object names are {object_names}. The response format is same as before. Can you give me some options to choose from using only the object name I gave you?"
        response = model.chat(query_text=query_text, image_paths=None, images=[rgb_img],
                                    PRINT_USER_MSG=True,
                                    PRINT_GPT_OUTPUT=True,
                                    RESET_CHAT=RESET_CHAT,
                                    RETURN_RESPONSE=True,
                                    DETAIL='high')
    response_json, error_message = response_to_json(response)

    #%%
    # 2.1 Visualize: source objects of the options
    extracted_args = extract_arguments(response_json)
    visualize_object_dict = {"obj_name": [], "obj_position": []}

    for args_idx, args in enumerate(extracted_args):
        # Check for necessary options; none-action / done-action
        if args == ['None', 'None'] or args == ['set_done()']:
            continue  # Skip the rest of the loop for these options

        # Iterate over each element in args
        print("[Visualize Options]: Extracted arguments:", args)
        for arg in args:
            if arg in env.body_names:
                p_object = env.get_p_body(body_name=arg)
                formatted_p_object = [f"{coord:.3f}" for coord in p_object]
            else:
                arg = input(f"Please input the object name manually for {arg}: ")
                if arg in env.body_names:
                    p_object = env.get_p_body(body_name=arg)
                    formatted_p_object = [f"{coord:.3f}" for coord in p_object]
                    print(f"{arg}: {formatted_p_object}")
                else:
                    print(f"{arg} not found in environment.")
                
            visualize_object_dict["obj_name"].append(arg)
            visualize_object_dict["obj_position"].append(p_object)

    # 2.2. TODO: visualize the target objects
    # # Get images
    # env.set_viewer(VERBOSE=False)

    # for _ in range(100): # for loop to place the object
    #     env.step(ctrl=np.append(init_pose,1.0),ctrl_idxs=idxs_ur_step+[6])
    #     [env.plot_sphere(p=obj_position+np.array([0,0,0.1]),r=0.01,rgba=env.model.geom(env.model.body(obj_name).geomadr[0]).rgba,label=f'src: {obj_name}') for obj_name, obj_position in zip(visualize_object_dict["obj_name"], visualize_object_dict["obj_position"])]
    #     # for obj_name, obj_position in zip(visualize_object_dict["obj_name"], visualize_object_dict["obj_position"]):
    #     #     env.plot_sphere(p=obj_position+np.array([0,0,0.1]),r=0.01,rgba=env.model.geom(env.model.body(obj_name).geomadr[0]).rgba,label=f'src: {obj_name}')
    # scene_img_after,scene_depth_img_after = env.grab_rgb_depth_img_offscreen()
    # p_cam,R_cam = env.get_pR_body(body_name='ur_camera_center')
    # p_ego  = p_cam
    # p_trgt = p_cam + R_cam[:,2]
    # rgb_img_after,depth_img_after,pcd,xyz_img_after,xyz_img_world_after = env.get_egocentric_rgb_depth_pcd_offscreen(
    #     p_ego=p_ego,p_trgt=p_trgt,rsz_rate=10,fovy=45,BACKUP_AND_RESTORE_VIEW=True)

    # env.close_viewer()

    #%%
    # 3. User prompt: choose the option
    print("\033[94m [User prompt: choose the option] \033[00m")
    print("Please input your choice:")

    # Get images
    env.set_viewer(VERBOSE=False)

    while True:
        scene_img_after, rgb_img_after, depth_img_after = env.get_image_both_mode(body_name='ur_camera_center')
        if env.loop_every(HZ=20):
            env.render(render_every=1)

        # Plot world view and egocentric RGB and depth images
        closest_obj_pairs_img_before, closest_obj_pairs_names_before, closest_obj_pairs_indices_before = env.get_closest_pairs(object_names, TOP_VIEW=TOP_VIEW, min_distance=min_distance)
        if env.MODE == 'offscreen':
            show_images(scene_img_after, rgb_img_after, closest_obj_pairs_img_before, None, None, None, title="Select option from the following scene and RGB images")

        option_idx = int(input())
        selected_option_idx.append(option_idx)
        len_options.append(len(response_json['options']))

        if option_idx == 'stop()':
            print("Interaction is Done (\033[91mAnnotator called stop()\033[0m)")
            break
        if isinstance(option_idx, str):
            option_idx = int(option_idx)
        if option_idx > 0 and option_idx <= len(response_json["options"]):
            break
        else:
            print(f"Input a valid option number: \033[91m1 ~ {len(response_json['options'])}\033[0m")
    print(f"You choose: \033[91m{option_idx}\033[0m")
    type_conversion = {
        'cuboid': 'box', 'prism': 'box', 'parallelepiped': 'box',
        'bowl_blue_01': 'bowl_blue', 'bowl_green_01': 'bowl_green', 'bowl_red_01': 'bowl_red',
        'dish': 'bowl', 'plate': 'bowl', 'cylinder': 'cylinder',
        'cube': 'box', 'rectangular_prism': 'box', 'hexagonal_prism': 'box'
    }
    if option_idx == 'stop()':
        print("Interaction is Done (\033[91mAnnotator called stop()\033[0m)")
        history_actions.append(option_idx)
    else:
        executable_actions = parse_actions_to_executable_strings(response_json, option_idx, env)
        print(f"You choose: \033[92m{executable_actions}\033[0m")
        history_actions.append(executable_actions)
    env.close_viewer()

    #%%
    ## 3.1. Execute the function
    print("\033[94m [Execute the action] \033[00m")
    env.set_viewer(VERBOSE=True)
    for exec_function in executable_actions:
        print(f"Executing: \033[92m{exec_function}\033[0m")
        exec(exec_function)

        env.reset()
        env.forward(q=init_pose,joint_idxs=idxs_ur_fwd)
        # Get images
        for _ in range(1000): # for loop to place the object
            env.forward(q=init_pose,joint_idxs=idxs_ur_fwd)
            env.step(ctrl=np.append(init_pose,1.0),ctrl_idxs=idxs_ur_step+[6])
        scene_img_after,rgb_img_after,depth_img_after = env.get_image_both_mode(body_name='ur_camera_center')
        if env.loop_every(HZ=20):
            env.render(render_every=1)
    # Set the object configuration after the action.
    env.set_object_configuration(object_names=object_names)
    # Plot world view and egocentric RGB and depth images
    closest_obj_pairs_img_after, closest_obj_pairs_names_after, closest_obj_pairs_indices_after = env.get_closest_pairs(object_names, TOP_VIEW=TOP_VIEW, min_distance=min_distance)

    if env.MODE == 'offscreen':
        show_images(scene_img_after, rgb_img_after, closest_obj_pairs_img_after, None, None, None, title="Initial scene and RGB images")
    env.close_viewer()
    if not (exec_function == "env.move_object(None, None)"):
        rgb_img_list.append(rgb_img_after)
        scene_img_list.append(scene_img_after)

    #%%
    # 4. Interaction for a new image, and repeat the process
    while True:
        # Get Images
        env.set_viewer(VERBOSE=False)
        env.forward(q=init_pose,joint_idxs=env.idxs_forward)
        scene_img, rgb_img, depth_img = env.get_image_both_mode(body_name='ur_camera_center')
        env.close_viewer()

        print("\033[94m [Interaction for a new image, and repeat the process] \033[00m")
        if env.get_done():
            print("Interaction is Done (\033[91mset_done()\033[0m is called)")
            break

        # 4.1. Input new scene image
        if np.random.rand() < 0.5:
            query_text = "I will give you a image that shows the result of the selected action. You only respond to JSON format. Can you give me some options to choose what I prefer to do next? You ONLY respond in json."
        else:
            query_text = "I will give you a image that shows the result of the selected action. You only respond to JSON format. Give me some options based on previous actions. You ONLY respond in json."
        response = model.chat(query_text=query_text, image_paths=None, images=[rgb_img_after],
                                    PRINT_USER_MSG=True,
                                    PRINT_GPT_OUTPUT=True,
                                    RESET_CHAT=False,
                                    RETURN_RESPONSE=True,
                                    DETAIL="high")
        response_json, error_message = response_to_json(response)

        # 4.2. TODO: visualize the target objects
        # 4.3. User prompt: choose the option
        ## User prompt: choose the option
        print("Please input your choice:")
        ## Get images
        env.set_viewer(VERBOSE=False)
        while True:
            scene_img,rgb_img,depth_img = env.get_image_both_mode(body_name='ur_camera_center')
            # Plot world view and egocentric RGB and depth images
            closest_obj_pairs_img, closest_obj_pairs_names, closest_obj_pairs_indices = env.get_closest_pairs(object_names, TOP_VIEW=TOP_VIEW, min_distance=min_distance)
            if env.MODE == 'offscreen':
                show_images(scene_img, rgb_img, closest_obj_pairs_img, None, None, None, title="Select option from the following scene and RGB images")

            if option_idx == 'stop()':
                print("Interaction is Done (\033[91mAnnotator called stop()\033[0m)")
                break
            option_idx = int(input())
            selected_option_idx.append(option_idx)
            len_options.append(len(response_json['options']))
    
            
            if isinstance(option_idx, str):
                option_idx = int(option_idx)
            if option_idx > 0 and option_idx <= len(response_json["options"]):
                break
            else:
                print(f"Input a valid option number: \033[91m1 ~ {len(response_json['options'])}\033[0m")

        print(f"You choose: \033[91m{option_idx}\033[0m")
        if option_idx == 'stop()':
            print("Interaction is Done (\033[91mAnnotator called stop()\033[0m)")
            history_actions.append(option_idx)
        else:
            executable_actions = parse_actions_to_executable_strings(response_json, option_idx, env)
            history_actions.append(executable_actions)
        print(f"You choose: \033[92m{executable_actions}\033[0m")
        env.close_viewer()

        ## Execute the function
        env.set_viewer(VERBOSE=False)
        for exec_function in executable_actions:
            print(f"Executing: \033[92m{exec_function}\033[0m")
            exec(exec_function)

            # Get images
            env.reset()
            for _ in range(1000): # for loop to place the object
                env.forward(q=init_pose,joint_idxs=idxs_ur_fwd)
                env.step(ctrl=np.append(init_pose,1.0),ctrl_idxs=idxs_ur_step+[6])
            scene_img_after,rgb_img_after,depth_img_after = env.get_image_both_mode(body_name='ur_camera_center')
            if env.loop_every(HZ=20):
                env.render(render_every=1)

        # Set the object configuration after the action.
        env.set_object_configuration(object_names=object_names)
        # Plot world view and egocentric RGB and depth images
        closest_obj_pairs_img_after, closest_obj_pairs_names_after, closest_obj_pairs_indices_after = env.get_closest_pairs(object_names, TOP_VIEW=TOP_VIEW, min_distance=min_distance)
        if env.MODE == 'offscreen' and not (exec_function == "env.move_object(None, None)"):
            show_images(scene_img, rgb_img, closest_obj_pairs_img, scene_img_after, rgb_img_after, closest_obj_pairs_img_after)

        env.close_viewer()
        if not (exec_function == "env.move_object(None, None)"):
            rgb_img_list.append(rgb_img_after)
            scene_img_list.append(scene_img_after)

    #%%
    # 5. End of the Interaction; Summary of the interaction and Reset the environment.
    print("\033[94m [End of the Interaction; Summary of the interaction and Reset the environment.] \033[00m")
    with open('./prompt/reason_preference.txt', 'r') as file:
        reason_preference_text = file.read()
    model.set_common_prompt(reason_preference_text)

    # query_text = f"The {make_ordinal(interaction_idx+1)} interaction is done. I will give you the changes in the scene through the interaction. Please summarize the interaction focusing on the preference of the user?"
    query_text = f"""The {make_ordinal(interaction_idx+1)} interaction is done. Concentrate on the changes in the scene through the interaction.
    Let's try to reason about the user's preference in the set of preference categories: place with same color, place with same shape, and stack with same shape"""
    # response = model.chat(query_text=query_text, image_paths=None, images=rgb_img_list,
    response_preference = model.chat(query_text=query_text, image_paths=None, images=None,
                                PRINT_USER_MSG=True,
                                PRINT_GPT_OUTPUT=True,
                                RESET_CHAT=False,
                                RETURN_RESPONSE=True)
    preference_json, error_message = response_to_json(response_preference)
    preference_history.append(response_preference)

    # Save rgb_img_list
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.save(args.save_dir+f'rgb_img_list_{interaction_idx+1}.npy', rgb_img_list)

    annotator_info["preference"] = "put blocks in a same colored bowl"
    annotator_info["preference_description"] = response_preference
    annotator_info["option_length"] = len_options
    annotator_info["option_idxs"] = selected_option_idx
    annotator_info["executed_actions"] = history_actions
    # annotator_info["preference_score"] = preference_annotator
    # annotator_info["total_token"] = annotator_total_token
    # annotator_info["image_token"] = annotator_image_token
    #%%
    # 6. Evaluate Preference Accuracy with pre-defined criteria.
    total_token = model.get_total_token()
    image_token = model.get_image_token()
    model.reset_tokens()
    print("\033[94m [Evaluate Preference Accuracy with pre-defined criteria.] \033[00m")
    color_match_count, shape_match_count, stack_count = env.evaluate_preference(closest_obj_pairs_names_after)
    object_positions = []
    for object_name in object_names:
        p_object = env.get_p_body(body_name=object_name)
        object_positions.append(p_object)
    stacked_order = get_stacked_order(object_positions, object_names)
    print(f"closest_obj_pairs_before: {closest_obj_pairs_names_after}")
    print("Color Match Count:", color_match_count)
    print("Shape Match Count:", shape_match_count)
    print("Stack Count:", stack_count)
    print("Stacked Object Order:", stacked_order)

    preference_score, preference = preference_criteria(color_match_count, shape_match_count, stack_count, stacked_order)
    interaction_info["preference_info"] = {}
    interaction_info["preference_info"]["color_match_count"] = color_match_count
    interaction_info["preference_info"]["shape_match_count"] = shape_match_count
    interaction_info["preference_info"]["stack_count"] = stack_count
    interaction_info["preference_info"]["stacked_order"] = stacked_order
    interaction_info["episode"] = interaction_idx+1
    interaction_info["num_interaction"] = len(history_actions)
    interaction_info["object_info"] = visualize_object_dict
    interaction_info["history_actions"] = history_actions
    selected_user_options = copy.copy(history_actions)
    selected_user_option_history.append(selected_user_options)
    interaction_info["preference_score"] = preference_score
    interaction_info["preference"] = preference
    interaction_info["total_token"] = total_token
    interaction_info["image_token"] = image_token
    print(f"User Preference: \033[92m{preference}\033[0m")

    # Save interaction history to a file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(args.save_dir+f'annotator_info_{interaction_idx+1}.json', 'w') as file:
        json.dump(annotator_info, file, indent=4)
    # save dictionary to a file
    with open(args.save_dir+f'interaction_info_{interaction_idx+1}.json', 'w') as file:
        json.dump(interaction_info, file, indent=4)

    # Save interaction history to a file
    model.save_interaction(data=model.messages, file_path=args.save_dir+f'interaction_{interaction_idx+1}.json')

#%%
# 7. End of the interaction loop
print("\033[94m [End of the interaction loop] \033[00m")
if env.is_viewer_alive():
    env.close()
print("Done.")

# Save Whole interaction history to a file
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
model.save_interaction(data=model.messages, file_path=args.save_dir+f'interaction_all.json')