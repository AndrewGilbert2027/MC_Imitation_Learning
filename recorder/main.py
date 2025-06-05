import minerl
import gym
import cv2
import json
import pygame
import numpy as np
# import torch
# import pickle
import os

# This code has been adapted from https://github.com/Infatoshi/vpt.git

samples = 1  # Number of samples to record
output_path = os.getcwd()

for i in range(39, samples+39):
    # Initialize pygame
    pygame.init()

    frames = []

    # Constants
    OUTPUT_VIDEO_FILE = f"{output_path}/data/labeller-training/mc-{i}.mp4"
    ACTION_LOG_FILE = f"{output_path}/data/labeller-training/mc-{i}.json"
    FPS = 30
    RESOLUTION = (640, 360)  # Resolution of the video (in pygame coordinates)
    OUTPUT_RESOLUTION = (256, 256) # Resolution of the output video (in OpenCV coordinates) (for network training)
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption('Minecraft')
    SENS = 0.05

    # Check if the output directory exists, if not create it (same as action log)
    output_dir = os.path.dirname(OUTPUT_VIDEO_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    # Set up the OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, FPS, OUTPUT_RESOLUTION, isColor=True)  # Ensure isColor=True for RGB

    if not out.isOpened():
        print(f"Error: Could not initialize video writer for '{OUTPUT_VIDEO_FILE}'")
        pygame.quit()
        exit()

    pygame.mouse.set_visible(False)
    pygame.mouse.set_pos(screen.get_width() // 2, screen.get_height() // 2)  # Center the mouse
    pygame.event.set_grab(True)

    prev_mouse_x, prev_mouse_y = pygame.mouse.get_pos()

    # Mapping from pygame key to action
    # (Will only need forward, back, left, right, jump, sprint)
    key_to_action_mapping = {
        pygame.K_w: {'forward': 1},
        pygame.K_s: {'back': 1},
        pygame.K_a: {'left': 1},
        pygame.K_d: {'right': 1},
        pygame.K_SPACE: {'jump': 1},
        pygame.K_LSHIFT: {'sprint': 1},
        }
        # ... movement keys, jump, crouch, sprint, hotbar, attack, use, inventory, drop, swaphands, pickitem
    # Mapping from mouse button to action (only need attack for this project)
    mouse_to_action_mapping = {
        0: {'attack': 1}     # Left mouse button
    }
    
    action_log = []

    # Initialize the Minecraft environment
    env_name = 'MineRLBasaltFindCave-v0'
    env = gym.make(env_name) 


    env.seed(2143)
    obs = env.reset()
   
    done = False
    """
    # action_space = {"ESC": 0,
    #          "noop": [], 
    #          "attack": 0, 
    #          "back": 0, 
    #          "forward": 0, 
    #          "jump": 0, 
    #          "left": 0, 
    #          "right": 0, 
    #          "sprint": 0, 
    #          "camera": [0.0, 0.0]}
    """
    try:
        while not done:
            # Get the current observation (pixel array)
            image = np.array(obs['pov'])
            
            # Resize the image to output resolution
            out_image = cv2.resize(image, OUTPUT_RESOLUTION)

            # Debugging: Check frame shape
            print(f"Frame shape: {out_image.shape}")

            # Write the frame to the video file
            out.write(out_image)
            print(f"Writing frame {len(frames)} to video file.")

            # Convert the image to a format suitable for pygame (reversed and rotated)
            image = np.flip(image, axis=1)
            image = np.rot90(image)
            image = pygame.surfarray.make_surface(image)
            screen.blit(image, (0, 0))
            pygame.display.update()
        
            # Get the current state of all keys
            keys = pygame.key.get_pressed()
        
            # Create an action dictionary (initially with 'noop')
            action = {'noop': []}
            for key, act in key_to_action_mapping.items():
                if keys[key]:
                    action.update(act)
            
            # Get mouse button states
            mouse_buttons = pygame.mouse.get_pressed()
            for idx, pressed in enumerate(mouse_buttons):
                if pressed:
                    action.update(mouse_to_action_mapping.get(idx, {}))
            
            # Get mouse movement
            mouse_x, mouse_y = pygame.mouse.get_pos()
            delta_x = mouse_x - prev_mouse_x
            delta_y = mouse_y - prev_mouse_y
        
            # Reset mouse to the center of the window 
            pygame.mouse.set_pos(screen.get_width() // 2, screen.get_height() // 2)
            prev_mouse_x, prev_mouse_y = screen.get_width() // 2, screen.get_height() // 2
        
            # Now, use delta_x and delta_y for the camera movement (sensitivity adjusted)
            action['camera'] = [delta_y * SENS, delta_x * SENS]
            print(f"Camera movement: {action['camera']}")
        
            # Add the in-game 'ESC' action to the beginning of the action
            action = {'ESC': 0, **action}
            action_log.append(action)
        
            # Apply the action in the environment
            obs, reward, done, _ = env.step(action)
        
            # Check if the 'q' key is pressed to terminate the loop
            if keys[pygame.K_q]:
                break
        
            # Handle pygame events to avoid the window becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
    except KeyboardInterrupt:
        pass
    finally:    
        # Cleanup
        out.release()
        pygame.quit()

        # Verify video file creation
        if os.path.exists(OUTPUT_VIDEO_FILE):
            print(f"Video file saved successfully: {OUTPUT_VIDEO_FILE}")
        else:
            print(f"Error: Video file '{OUTPUT_VIDEO_FILE}' was not saved.")
        
        # Save the actions to a JSON file
        if not os.path.exists(os.path.dirname(ACTION_LOG_FILE)):
            os.makedirs(os.path.dirname(ACTION_LOG_FILE))
        with open(ACTION_LOG_FILE, 'w') as f:
            # write to jsonl
            f.write(json.dumps(action_log))

    cv2.namedWindow('Recorded Video', cv2.WINDOW_AUTOSIZE)
    cap = cv2.VideoCapture(OUTPUT_VIDEO_FILE)

    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        # Display the recorder video (sanity check)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Recorded Video', frame)
            
            # Adjust the delay for cv2.waitKey() if the playback speed is not correct
            if cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

