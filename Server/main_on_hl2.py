import sys, os
import asyncio
import time
from collections import deque
import cv2
import numpy as np
import websockets
import struct
import json
from PIL import Image, ImageDraw, ImageFont

from modules import HandTracker_our, HandTracker_our_v2, GestureClassfier, ObjTracker
from collections import deque
from utils.visualize import draw_2d_skeleton

sys.path.append("./hl2ss_")
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_utilities
import socket
import multiprocessing as mp
import keyboard


## args ##
ckpt = "checkpoint-40.tar"
flag_interaction_detect = False

## Set HoloLens2 wifi address ##
host = '192.168.50.31'


# Calibration path (must exist but can be empty)
calibration_path = 'calibration'

# Front RGB camera parameters
pv_width = 640      # (1080, 1920), (720, 1280), (360, 640), (240, 424)
pv_height = 360
pv_fps = 30

# Buffer length in seconds
buffer_size = 10

# process depth image per n frame
num_depth_count = 10    # 0 for only rgb

# gesture sequence args
seq_len = 16
threshold_num = 5


prev = time.time()
prev_label = "init"

def main():
    ###################### init models ######################

    track_hand_v1 = HandTracker_our()
    track_hand_v2 = HandTracker_our_v2()
    flag_hand_model = True

    track_gesture = GestureClassfier(ckpt=f"./gestureclassifier/checkpoints/{ckpt}", seq_len=seq_len, model_opt=1)

    if flag_interaction_detect:
        track_obj = ObjTracker()

    ###################### init comm. with hololens2 ######################
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    init_variables, max_depth, producer = init_hl2()

    cv2.namedWindow('Prompt')
    cv2.resizeWindow(winname='Prompt', width=500, height=500)
    cv2.moveWindow(winname='Prompt', x=2000, y=200)

    idx_depth = 0
    idx = 0

    gesture_idx = -1
    flag_cooldown = False
    t_cooldown = 0.0

    queue_righthand = deque([], maxlen=seq_len)
    flag_gesture = False

    prev_gesture = None
    gesture_cnt = 0
    valid_gesture = None
    valid_gesture_idx = -1

    debug_pose = np.ones((21, 3))

    log_t = deque([], maxlen=50)
    t1 = 0
    try:
        while True:
            if flag_hand_model:
                track_hand = track_hand_v2
            else:
                track_hand = track_hand_v1

            if keyboard.is_pressed('space'):
                flag_hand_model = not flag_hand_model

            idx+=1

            # intermittently receive depth image
            idx_depth += 1
            if idx_depth == num_depth_count:
                idx_depth = 0
                flag_depth = True
            else:
                flag_depth = False

            # log_t.append(time.time() - t1)
            # if len(log_t) > 10:
            #     print(np.average(np.array(log_t)))
            ###################### receive input ######################
            result = receive_images(init_variables, flag_depth)
            if result == None:
                continue
            # t1 = time.time()
            color, depth = result

            ### Display RGBD pair ###
            cv2.imshow('RGB', color)
            if flag_depth:
                cv2.imshow('Depth', depth / max_depth)  # scale for visibility
            cv2.waitKey(1)

            # print((time.time() - t1)*1000)
            # continue

            color = cv2.resize(color, dsize=(640, 360), interpolation=cv2.INTER_AREA)

            ###################### process hand ######################

            outs = track_hand.run(np.copy(color))   # uvd_right. return only right hand when visible
            if not isinstance(outs, np.ndarray):
                continue


            ###################### process gesture ######################
            # preprocess joint pose
            angle_label = track_gesture._compute_ang_from_joint(outs)
            data = np.concatenate([outs.flatten(), angle_label])
            queue_righthand.append(data)

            ###################### interaction prediction ######################
            """
            - 손 위치 depth로부터 10cm 내에서 detect된 물체 boundingbox만 체크
            - palm 2D pose로부터 10cm 내에 물체 bb가 있으면 gesture recognizer 활성화. 
            """

            if flag_depth and flag_interaction_detect:

                uv_wrist = outs[0, :-1]

                if int(uv_wrist[1]) > pv_height or int(uv_wrist[0]) > pv_width:
                    flag_gesture = False
                else:
                    d_wrist = depth[int(uv_wrist[1]), int(uv_wrist[0])]

                    palm_points_2d = outs[[0, 4, 8, 12, 16, 20], :2]
                    palm_uv = np.mean(palm_points_2d, axis=0)

                    obj_nearby_list = track_obj.detect_objs_no_cnt(color, depth, d_wrist)

                    # 손 근처 물체 BB 시각화.
                    if len(obj_nearby_list) > 0:
                        vis_obj = color.copy()
                        for obj_nearby in obj_nearby_list:
                            # draw every nearby object bb in green
                            cv2.rectangle(vis_obj, (obj_nearby[0], obj_nearby[1]), (obj_nearby[2], obj_nearby[3]),
                                          (0, 255, 0), 2)
                            cv2.putText(vis_obj, f'{obj_nearby[-1]}', (obj_nearby[0], obj_nearby[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        ## activate gesture recognizer ##
                        for obj_nearby in obj_nearby_list:
                            cx, cy = (obj_nearby[0] + obj_nearby[2]) // 2, (obj_nearby[1] + obj_nearby[3]) // 2
                            distance = np.sqrt((cx - palm_uv[0]) ** 2 + (cy - palm_uv[1]) ** 2)

                            if distance < 70.0:
                                # draw object bb if it's close to hand, as yellow
                                cv2.rectangle(vis_obj, (obj_nearby[0], obj_nearby[1]), (obj_nearby[2], obj_nearby[3]),
                                              (0, 255, 255), 2)
                                cv2.putText(vis_obj, f'{obj_nearby[-1]}', (obj_nearby[0], obj_nearby[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                flag_gesture = True

                                # if any of object are close to hand, flag_gesture is True and no need to iterate
                                break
                            else:
                                flag_gesture = False

                        cv2.imshow("obj", vis_obj)
                    # elif track_obj.flag_detected:  # off the gesture only when no nearby bb has found from obj detection
                    #     flag_gesture = False


            if flag_gesture or not flag_interaction_detect:
                if len(queue_righthand) < seq_len:
                    continue

                # t1 = time.time()
                gesture_idx, gesture = track_gesture.run(queue_righthand)  # queue (10, 63+15)
                # log_t.append(time.time() - t1)
                #
                # if len(log_t) > 100:
                #     print("avg : ", np.average(np.array(log_t)))
            else:
                gesture = None

            ## valid gesture if same gesture continously detected
            if prev_gesture == gesture and gesture != 'Natural':
                gesture_cnt += 1
            else:
                gesture_cnt = 0
            prev_gesture = gesture

            if gesture_cnt > threshold_num:
                valid_gesture = gesture
                valid_gesture_idx = gesture_idx

            ###################### visualize ######################
            color = draw_2d_skeleton(color, outs[:, :2])

            if valid_gesture != None and valid_gesture != "Natural":
                cv2.putText(color, f'{valid_gesture.upper()}',
                            org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255),
                            thickness=3)

            cv2.imshow("Prompt", color)
            cv2.waitKey(1)


            ###################### send to hololens2 ######################
            ## check cooldown. 0.5 sec delay for each gesture
            if time.time() - t_cooldown > 0.5:
                flag_cooldown = False

            if not flag_cooldown and valid_gesture != None and valid_gesture != "Natural":

                send_data = outs.flatten().tolist() + [float(valid_gesture_idx), float(time.time()*1000)]

                flag_cooldown = True
                t_cooldown = time.time()
                print("sending ... ", valid_gesture)
            else:
                # dummy = np.asarray([debug_idx, float(-1)], dtype=np.float64)
                # send_data = ["None", str(time.time()*1000)]
                debug_pose = np.ones((63))

                send_data = debug_pose.tolist() + [float(-1), float(time.time()*1000)]

            fmt = f"{len(send_data)}d"
            send_bytes = struct.pack(fmt, *send_data)

            sock.sendto(send_bytes, (host, 5005))

    finally:
        sock.close()

        # Stop PV and RM Depth AHAT streams ---------------------------------------
        sink_ht, sink_pv = init_variables[0], init_variables[1]
        sink_pv.detach()
        sink_ht.detach()
        producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
        producer.stop(hl2ss.StreamPort.RM_DEPTH_AHAT)

        # Stop PV subsystem -------------------------------------------------------
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        cv2.destroyAllWindows()


def init_hl2():
    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Get RM Depth AHAT calibration -------------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_ht = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_AHAT, calibration_path)

    uv2xy = calibration_ht.uv2xy  # hl2ss_3dcv.compute_uv2xy(calibration_ht.intrinsics, hl2ss.Parameters_RM_DEPTH_AHAT.WIDTH, hl2ss.Parameters_RM_DEPTH_AHAT.HEIGHT)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_ht.scale)
    max_depth = calibration_ht.alias / calibration_ht.scale

    xy1_o = hl2ss_3dcv.block_to_list(xy1[:-1, :-1, :])
    xy1_d = hl2ss_3dcv.block_to_list(xy1[1:, 1:, :])

    # Start PV and RM Depth AHAT streams --------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO,
                       hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height,
                                       framerate=pv_fps))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_fps * buffer_size)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss.Parameters_RM_DEPTH_AHAT.FPS * buffer_size)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.RM_DEPTH_AHAT)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_ht = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_AHAT, manager, None)

    sink_pv.get_attach_response()
    sink_ht.get_attach_response()

    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    return [sink_ht, sink_pv, pv_intrinsics, pv_extrinsics, xy1_o, xy1_d, scale, calibration_ht], max_depth, producer


def receive_images(init_variables, flag_depth):

    sink_ht, sink_pv, pv_intrinsics, pv_extrinsics, xy1_o, xy1_d, scale, calibration_ht = init_variables

    # Get RM Depth AHAT frame and nearest (in time) PV frame --------------
    _, data_ht = sink_ht.get_most_recent_frame()
    if ((data_ht is None) or (not hl2ss.is_valid_pose(data_ht.pose))):
        return None
    _, data_pv = sink_pv.get_nearest(data_ht.timestamp)
    if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
        return None

    # Preprocess frames ---------------------------------------------------
    color = data_pv.payload.image

    pv_z = None
    if flag_depth:
        depth = data_ht.payload.depth  # hl2ss_3dcv.rm_depth_undistort(data_ht.payload.depth, calibration_ht.undistort_map)
        z = hl2ss_3dcv.rm_depth_normalize(depth, scale)

    # Update PV intrinsics ------------------------------------------------
    # PV intrinsics may change between frames due to autofocus
    pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length,
                                               data_pv.payload.principal_point)
    color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

    # Generate depth map for PV image -------------------------------------
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
        pv_z = np.zeros((pv_height, pv_width), dtype=np.float32)

        u0 = pv_list[:, 0]
        v0 = pv_list[:, 1]
        u1 = pv_list[:, 2]
        v1 = pv_list[:, 3]

        mask0 = (u0 >= 0) & (u0 < pv_width) & (v0 >= 0) & (v0 < pv_height)
        mask1 = (u1 > 0) & (u1 <= pv_width) & (v1 > 0) & (v1 <= pv_height)
        maskf = mask0 & mask1

        pv_list = pv_list[maskf, :]
        pv_list_depth = pv_list_depth[maskf, 0]

        for n in range(0, pv_list.shape[0]):
            u0 = pv_list[n, 0]
            v0 = pv_list[n, 1]
            u1 = pv_list[n, 2]
            v1 = pv_list[n, 3]

            pv_z[v0:v1, u0:u1] = pv_list_depth[n]

    return color, pv_z


def log_event(label, flag_time=False):
    global prev_label, prev

    now = time.time()
    if flag_time:
        print(f"{prev_label} ~ {label}: {now - prev:.3f}")
    prev_label = label
    prev = now

if __name__ == '__main__':
    main()