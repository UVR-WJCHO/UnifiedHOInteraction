import copy
import sys
import os
import time
import struct
import socket
import multiprocessing as mp
from collections import deque

import cv2
import numpy as np
import keyboard
from PIL import Image, ImageDraw, ImageFont

# Local modules
from modules import HandTracker, HandTracker_v2, GestureClassfier, ObjTracker
from utils.visualize import draw_2d_skeleton

# HL2SS modules
sys.path.append("./hl2ss_")
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_utilities

# ==============================================================================
# 1. Configuration Constants
# ==============================================================================

# flags
flag_onHololens2 = True
flag_hand_model = True  # Default to v2
flag_save = False
flag_vis = True

# HoloLens 2 Connection Settings
HOST_ADDRESS = '192.168.0.112'
CALIBRATION_PATH = 'calibration'

# Front RGB Camera Parameters
PV_WIDTH = 640
PV_HEIGHT = 360
PV_FPS = 30

# Buffer settings
BUFFER_SIZE = 10

# Depth processing interval (process every N frames)
NUM_DEPTH_COUNT = 10

# Gesture Sequence Settings
SEQ_LEN = 16
THRESHOLD_NUM = 5

# Gesture Model Parameters
CKPT_NAME = "checkpoint-40.tar"     # handtracker/checkpoint/{...}/{CKPT_NAME}


# ==============================================================================
# 2. Main
# ==============================================================================

def main():
    global flag_onHololens2, flag_hand_model, flag_save, flag_vis

    # --- Initialize Models ---
    track_hand_v2 = HandTracker_v2()
    track_hand_v1 = HandTracker()

    gesture_ckpt_path = f"./gestureclassifier/checkpoints/{CKPT_NAME}"
    track_gesture = GestureClassfier(ckpt=gesture_ckpt_path, seq_len=SEQ_LEN, model_opt=1)

    track_obj = ObjTracker()

    if flag_onHololens2:
        # --- Initialize Communication with HoloLens 2 ---
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        init_variables, max_depth, producer = init_hl2()
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("no webcam input")
            exit()

    # --- UI Setup ---
    cv2.namedWindow('Prompt')
    cv2.resizeWindow(winname='Prompt', width=500, height=500)
    cv2.moveWindow(winname='Prompt', x=2000, y=200)

    # --- Loop Variables ---
    idx_depth = 0
    idx = 0
    flag_cooldown = False
    t_cooldown = 0.0

    queue_righthand = deque([], maxlen=SEQ_LEN)
    prev_gesture = None
    gesture_cnt = 0
    valid_gesture = None
    valid_gesture_idx = -1

    keyboard.on_press_key("space", toggle_hand_model)

    try:
        while True:
            idx += 1
            # Switch hand tracking model
            if flag_hand_model:
                track_hand = track_hand_v2
            else:
                track_hand = track_hand_v1

            # Periodically receive depth image
            idx_depth += 1
            if idx_depth == NUM_DEPTH_COUNT:
                idx_depth = 0
                flag_depth = True
            else:
                flag_depth = False

            # Receive Image Input from HoloLens 2
            if flag_onHololens2:
                result = receive_images(init_variables, flag_depth)
                if result is None:
                    continue
                color, depth = result
            # Webcam input
            else:
                ret, frame = cap.read()
                color = frame
                depth = None
                flag_depth = False

            # Compute normalized depth if needed
            depth_norm = None
            if (flag_depth and flag_onHololens2) and (flag_vis or flag_save):
                depth_norm = ((depth * 255.0) / max_depth).astype('uint8')

            # Display
            if flag_vis:
                cv2.imshow('RGB', color)
                if depth_norm is not None:
                    cv2.imshow('Depth', depth_norm)
                cv2.waitKey(1)
            if flag_save:
                cv2.imwrite(f"./output/rgb/rgb_{idx}.png", color)
                if depth_norm is not None:
                    cv2.imwrite(f"./output/depth/depth_{idx}.png", depth_norm)

            # --- Process Hand Tracking ---
            outs = track_hand.run(np.copy(color), cnt=idx)  # Currently returns uvd_right (right hand only)
            if not isinstance(outs, np.ndarray):
                continue

            # --- Process Gesture ---
            # Preprocess joint pose for gesture classification
            outs_copy = copy.deepcopy(outs)
            angle_label = track_gesture._compute_ang_from_joint(outs)
            data = np.concatenate([outs.flatten(), angle_label])
            queue_righthand.append(data)

            # --- Interaction Prediction (Optional) ---
            # Logic: Check object bounding boxes within 10cm of the hand in depth
            # If an object is within 10cm of the palm in 2D space, activate gesture recognizer.
            flag_gesture = False
            if flag_depth:
                uv_wrist = outs[0, :-1]

                if int(uv_wrist[1]) >= PV_HEIGHT or int(uv_wrist[0]) >= PV_WIDTH:
                    flag_gesture = False
                else:
                    d_wrist = depth[int(uv_wrist[1]), int(uv_wrist[0])]

                    # Calculate palm center
                    palm_points_2d = outs[[0, 4, 8, 12, 16, 20], :2]
                    palm_uv = np.mean(palm_points_2d, axis=0)

                    # Use ObjTracker without internal counter
                    obj_nearby_list = track_obj.detect_objs_no_cnt(color, depth, d_wrist)

                    # Visualize nearby objects
                    if len(obj_nearby_list) > 0:
                        vis_obj = color.copy()
                        for obj_nearby in obj_nearby_list:
                            # Draw bounding box in green
                            cv2.rectangle(vis_obj, (obj_nearby[0], obj_nearby[1]), (obj_nearby[2], obj_nearby[3]),
                                          (0, 255, 0), 2)
                            cv2.putText(vis_obj, f'{obj_nearby[-1]}', (obj_nearby[0], obj_nearby[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # Activate gesture recognizer if hand is close to an object
                        for obj_nearby in obj_nearby_list:
                            cx, cy = (obj_nearby[0] + obj_nearby[2]) // 2, (obj_nearby[1] + obj_nearby[3]) // 2
                            distance = np.sqrt((cx - palm_uv[0]) ** 2 + (cy - palm_uv[1]) ** 2)

                            if distance < 70.0:
                                # Draw bounding box in yellow
                                cv2.rectangle(vis_obj, (obj_nearby[0], obj_nearby[1]), (obj_nearby[2], obj_nearby[3]),
                                              (0, 255, 255), 2)
                                cv2.putText(vis_obj, f'{obj_nearby[-1]}', (obj_nearby[0], obj_nearby[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                flag_gesture = True
                                break  # Stop after finding the first close object
                            else:
                                flag_gesture = False

                        if flag_vis:
                            cv2.imshow("obj", vis_obj)
                        if flag_save:
                            cv2.imwrite(os.path.join("./output", "obj", f"obj_{idx}.png"), vis_obj)
            else:
                flag_gesture = True

            # --- Run Gesture Recognition ---
            # Run if interaction flag is set OR interaction detection is disabled
            gesture = None
            gesture_idx = -1
            if flag_gesture:
                if len(queue_righthand) >= SEQ_LEN:
                    gesture_idx, gesture = track_gesture.run(queue_righthand)


            # Validate gesture (must be detected consistently)
            if prev_gesture == gesture and gesture != 'Natural':
                gesture_cnt += 1
            else:
                gesture_cnt = 0
            prev_gesture = gesture

            if gesture_cnt > THRESHOLD_NUM:
                valid_gesture = gesture
                valid_gesture_idx = gesture_idx

            # --- Visualize Skeleton and Gesture ---
            color = draw_2d_skeleton(color, outs_copy[:, :2])
            if flag_save:
                cv2.imwrite(f"./output/joints/joints_{idx}.png", color)

            if valid_gesture is not None and valid_gesture != "Natural":
                cv2.putText(color, f'{valid_gesture.upper()}',
                            org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255),
                            thickness=3)

            cv2.imshow("Prompt", color)
            cv2.waitKey(1)

            # --- Send Data to HoloLens 2 ---
            if flag_onHololens2:
                # Check cooldown (0.5 sec delay)
                if time.time() - t_cooldown > 0.0:
                    flag_cooldown = False
                if not flag_cooldown and valid_gesture is not None and valid_gesture != "Natural":
                    send_data = outs_copy.flatten().tolist() + [float(valid_gesture_idx), 0.0] #float(time.time() * 1000)]
                    flag_cooldown = True
                    t_cooldown = time.time()
                    print(f"sending ... {valid_gesture_idx}: {valid_gesture}")

                    fmt = f"{len(send_data)}d"
                    send_bytes = struct.pack(fmt, *send_data)

                    sock.sendto(send_bytes, (HOST_ADDRESS, 5005))

                    # reset after sending gesture
                    valid_gesture = None
                    valid_gesture_idx = -1
                    queue_righthand.clear()
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        if flag_onHololens2:
            sock.close()
            # Stop streams
            sink_ht, sink_pv = init_variables[0], init_variables[1]
            sink_pv.detach()
            sink_ht.detach()
            producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
            producer.stop(hl2ss.StreamPort.RM_DEPTH_AHAT)
            hl2ss_lnm.stop_subsystem_pv(HOST_ADDRESS, hl2ss.StreamPort.PERSONAL_VIDEO)



def init_hl2():
    # Start PV Subsystem
    hl2ss_lnm.start_subsystem_pv(HOST_ADDRESS, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Get Calibration Data
    calibration_ht = hl2ss_3dcv.get_calibration_rm(HOST_ADDRESS, hl2ss.StreamPort.RM_DEPTH_AHAT, CALIBRATION_PATH)

    uv2xy = calibration_ht.uv2xy
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_ht.scale)
    max_depth = calibration_ht.alias / calibration_ht.scale

    xy1_o = hl2ss_3dcv.block_to_list(xy1[:-1, :-1, :])
    xy1_d = hl2ss_3dcv.block_to_list(xy1[1:, 1:, :])

    # Start Streams
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO,
                       hl2ss_lnm.rx_pv(HOST_ADDRESS, hl2ss.StreamPort.PERSONAL_VIDEO, width=PV_WIDTH, height=PV_HEIGHT,
                                       framerate=PV_FPS))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT,
                       hl2ss_lnm.rx_rm_depth_ahat(HOST_ADDRESS, hl2ss.StreamPort.RM_DEPTH_AHAT))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, PV_FPS * BUFFER_SIZE)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss.Parameters_RM_DEPTH_AHAT.FPS * BUFFER_SIZE)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.RM_DEPTH_AHAT)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_ht = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_AHAT, manager, None)

    sink_pv.get_attach_response()
    sink_ht.get_attach_response()

    # Initialize PV intrinsics and extrinsics
    pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    return [sink_ht, sink_pv, pv_intrinsics, pv_extrinsics, xy1_o, xy1_d, scale, calibration_ht], max_depth, producer


def receive_images(init_variables, flag_depth):
    sink_ht, sink_pv, pv_intrinsics, pv_extrinsics, xy1_o, xy1_d, scale, calibration_ht = init_variables

    # Get frames
    _, data_ht = sink_ht.get_most_recent_frame()
    if ((data_ht is None) or (not hl2ss.is_valid_pose(data_ht.pose))):
        return None
    _, data_pv = sink_pv.get_nearest(data_ht.timestamp)
    if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
        return None

    # Preprocess frames
    color = data_pv.payload.image

    pv_z = None
    if flag_depth:
        depth = data_ht.payload.depth
        z = hl2ss_3dcv.rm_depth_normalize(depth, scale)

    # Update PV intrinsics (autofocus handling)
    pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length,
                                               data_pv.payload.principal_point)
    color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

    # Generate depth map for PV image
    if flag_depth:
        mask = (depth[:-1, :-1].reshape((-1,)) > 0)
        zv = hl2ss_3dcv.block_to_list(z[:-1, :-1, :])[mask, :]

        ht_to_pv_image = hl2ss_3dcv.camera_to_rignode(calibration_ht.extrinsics) @ hl2ss_3dcv.reference_to_world(
            data_ht.pose) @ hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(
            color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)

        ht_points_o = hl2ss_3dcv.rm_depth_to_points(xy1_o[mask, :], zv)
        pv_uv_o_h = hl2ss_3dcv.transform(ht_points_o, ht_to_pv_image)
        pv_list_depth = pv_uv_o_h[:, 2:]

        ht_points_d = hl2ss_3dcv.rm_depth_to_points(xy1_d[mask, :], zv)
        pv_uv_d_h = hl2ss_3dcv.transform(ht_points_d, ht_to_pv_image)
        pv_d_depth = pv_uv_d_h[:, 2:]

        mask = (pv_list_depth[:, 0] > 0) & (pv_d_depth[:, 0] > 0)

        pv_list_depth = pv_list_depth[mask, :]
        pv_d_depth = pv_d_depth[mask, :]

        pv_list_o = pv_uv_o_h[mask, 0:2] / pv_list_depth
        pv_list_d = pv_uv_d_h[mask, 0:2] / pv_d_depth

        pv_list = np.hstack((pv_list_o, pv_list_d + 1)).astype(np.int32)
        pv_z = np.zeros((PV_HEIGHT, PV_WIDTH), dtype=np.float32)

        # Mapping depth points to PV image coordinates
        u0, v0 = pv_list[:, 0], pv_list[:, 1]
        u1, v1 = pv_list[:, 2], pv_list[:, 3]

        mask0 = (u0 >= 0) & (u0 < PV_WIDTH) & (v0 >= 0) & (v0 < PV_HEIGHT)
        mask1 = (u1 > 0) & (u1 <= PV_WIDTH) & (v1 > 0) & (v1 <= PV_HEIGHT)
        maskf = mask0 & mask1

        pv_list = pv_list[maskf, :]
        pv_list_depth = pv_list_depth[maskf, 0]

        for n in range(0, pv_list.shape[0]):
            u0, v0 = pv_list[n, 0], pv_list[n, 1]
            u1, v1 = pv_list[n, 2], pv_list[n, 3]

            pv_z[v0:v1, u0:u1] = pv_list_depth[n]

    return color, pv_z


def toggle_hand_model(e):
    global flag_hand_model
    flag_hand_model = not flag_hand_model
    if flag_hand_model:
        name_hand_model = "WiLor"
    else:
        name_hand_model = "TEGCN"
    print(f"hand model switched: {name_hand_model}")

if __name__ == '__main__':
    main()