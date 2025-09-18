import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))     # append current dir to PATH
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../"))

import torch
import cv2
import time
import numpy as np
import copy

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO


import config as cfg
from wilor.utils.visualize import draw_2d_skeleton
from wilor.datasets.utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)
from skimage.filters import gaussian
import pyrealsense2 as rs


LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
prev = time.time()
prev_label = "init"
flag_gesture = False



## debug args ##
flag_webcam = False
flag_rsrecord = False
flag_time = False
flag_render_mesh = False

log_t_dict = {"start": [], "detection": [], "preprocess":[], "inference":[],"postprocess":[]}

if flag_rsrecord:
    pipeline = rs.pipeline()
    config = rs.config()

    ## from record
    bag_path = "C:/Woojin/research/realsense/record/20250821_181307.bag"  # color : 640 480, depth : 640 480
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
    # from realtime
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    colorizer = rs.colorizer()

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # playback 객체를 가져와서 재생 위치를 수동으로 조정
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)  # 실시간 재생 비활성화

    align_to = rs.stream.color
    align = rs.align(align_to)


class HandTracker_wilor():
    def __init__(self, img_w=640, img_h=360, det_cooltime=10):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(curr_dir, 'pretrained_models', 'wilor_final.ckpt')
        cfg_path = os.path.join(curr_dir, 'pretrained_models', 'model_config.yaml')
        mano_path = os.path.join(curr_dir, 'mano_data')

        YOLO_hand_path = os.path.join(curr_dir, 'pretrained_models', 'detector.pt')

        print("check input image scale. default : img_w=640, img_h=360")

        self.model, self.model_cfg = load_wilor(checkpoint_path=checkpoint_path, cfg_path=cfg_path, mano_path=mano_path)
        self.detector = YOLO(YOLO_hand_path)
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model.to(self.device)
        self.detector.to(self.device)
        self.model.eval()

        ## instead of dataset class, save configs
        # self.ViTdataset = ViTDetDataset(self.model_cfg)

        self.model_img_size = self.model_cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.model_cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.model_cfg.MODEL.IMAGE_STD)
        self.BBOX_SHAPE = self.model_cfg.MODEL.get('BBOX_SHAPE', None)

        self.img_w = img_w
        self.img_h = img_h

        ## update target bbox for every 10 frame, otherwise, update it following from previous pose
        self.det_cooltime = det_cooltime

        self.obj_cnt = 0
        self.bbox_cnt = 0
        self.bbox_togo = None
        self.pixelperframe = 2

        self.prev_uvd = []
        self.boxes = None
        self.right = None



        ## do first iteration
        log_event("start")
        testImg = cv2.imread(os.path.join(curr_dir, './demo_img/test1.jpg'))
        testImg = cv2.resize(testImg, (640, 360))

        log_event("load input")
        detections = self.detect_hands(testImg, conf=0.3, verbose=False)

        if not detections:
            print("no hand in inital img")
        else:
            boxes, right = detections

            batchs = self.preprocess(testImg, boxes, right, rescale_factor=2.0)
            log_event("preprocess data")

            with torch.no_grad():
                _ = self.model(batchs)
            log_event("done inference")

            img_size = batchs["img_size"].float()
            self.scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            print("success on first run")



    def run(self, img, flag_time=False):
        log_event("start", flag_time)
        if img.shape[-1] == 4:
            img = img[:, :, :-1]

        ## simple YOLO detection for every frame
        detections = self.detect_hands(img, conf=0.3, verbose=False)
        log_event("detection", flag_time)
        if not detections:
            return False

        self.boxes, self.right = detections

        batch = self.preprocess(img, self.boxes, self.right, rescale_factor=2.0)
        log_event("preprocess", flag_time)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints = []
        all_kpts = []
        all_uvds = []

        with torch.no_grad():
            out = self.model(batch)
        log_event("inference", flag_time)
        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam']
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                           self.scaled_focal_length).detach().cpu().numpy()

        # extract the result
        batch_size = batch['img'].shape[0]  # number of detected hands
        for n in range(batch_size):
            # Get filename from path img_path
            # img_fn, _ = os.path.splitext(os.path.basename(img_path))

            verts = out['pred_vertices'][n].detach().cpu().numpy()
            joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()

            is_right = batch['right'][n].cpu().numpy()
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
            joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
            cam_t = pred_cam_t_full[n]
            kpts_2d = project_full_img(verts, cam_t, self.scaled_focal_length, img_size[n])
            joints_2d = project_full_img(joints, cam_t, self.scaled_focal_length, img_size[n])

            uvd = np.zeros((21, 3))
            uvd[:, :2] = joints_2d
            uvd[:, -1] = joints[:, -1] * 1000.0  # depth in mm scale?

            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right)
            all_joints.append(joints)
            all_kpts.append(kpts_2d)
            all_uvds.append(uvd)

        self.prev_uvd = all_uvds.copy()
        log_event("postprocess", flag_time)

        ## process only right hand visible
        indices = np.where(np.asarray(all_right) == 1)[0]  ### check. 0: left, 1: right
        if len(indices) > 0:
            uvd_right = np.squeeze(np.asarray(all_uvds)[indices[0]])
            return uvd_right
        else:
            return False



    def preprocess(self, img_cv2: np.array,
                 boxes: np.array,
                 right: np.array, rescale_factor=2.5):

        boxes = boxes.astype(np.float32)
        centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        scales = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        personids = np.arange(len(boxes), dtype=np.int32)
        rights = right.astype(np.float32)

        num_data = boxes.shape[0]

        items = []
        for idx in range(num_data):
            center = centers[idx].copy()
            center_x = center[0]
            center_y = center[1]

            scale = scales[idx]
            bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=self.BBOX_SHAPE).max()

            patch_width = patch_height = self.model_img_size

            right = rights[idx].copy()
            flip = right == 0

            # 3. generate image patch
            # if use_skimage_antialias:
            cvimg = img_cv2.copy()
            if False:
                # Blur image to avoid aliasing artifacts
                downsampling_factor = ((bbox_size * 1.0) / patch_width)
                # print(f'{downsampling_factor=}')
                downsampling_factor = downsampling_factor / 2.0
                if downsampling_factor > 1.1:
                    cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

            img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                           center_x, center_y,
                                                           bbox_size, bbox_size,
                                                           patch_width, patch_height,
                                                           flip, 1.0, 0,
                                                           border_mode=cv2.BORDER_CONSTANT)
            img_patch_cv = img_patch_cv[:, :, ::-1]
            img_patch = convert_cvimg_to_tensor(img_patch_cv)

            # apply normalization
            for n_c in range(min(img_cv2.shape[2], 3)):
                img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

            item = {'img': torch.tensor(img_patch, device=self.device),
                    'personid': torch.tensor(int(personids[idx]), device=self.device),
                    'box_center': torch.tensor(centers[idx].copy(), device=self.device),
                    'box_size': torch.tensor(bbox_size, device=self.device),
                    'img_size': torch.tensor(1.0 * np.array([cvimg.shape[1], cvimg.shape[0]]), device=self.device),
                    'right': torch.tensor(rights[idx].copy(), device=self.device)
                    }

            items.append(item)

        merged = {}
        for key in items[0].keys():
            merged[key] = torch.stack([item[key] for item in items])

        return merged

    def detect_hands(self, img, conf, verbose):

        self.bbox_cnt += 1

        ## run YOLO when ... until the hand is detected on init & every cooltime done
        if self.boxes is None or self.bbox_cnt > self.det_cooltime:
            # print("running YOLO")

            self.bbox_cnt = 0

            detections = self.detector.predict(img, conf=conf, verbose=verbose, max_det=5)[0]

            bboxes = []
            is_right = []
            for det in detections:
                Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
                bboxes.append(Bbox[:4].tolist())

            if len(bboxes) == 0:
                return False

            boxes = np.stack(bboxes)
            right = np.stack(is_right)

        ## if not, update bbox position due to previous pose prediction
        else:
            new_bboxes = []

            for prev_pose, box, right in zip(self.prev_uvd, self.boxes, self.right):
                c_x, c_y = calc_pred_center(prev_pose)

                x1, y1, x2, y2 = np.squeeze(box.copy().astype(np.float32))
                w = x2 - x1
                h = y2 - y1

                # 새로운 중심 좌표 기준으로 bbox 좌표 계산
                new_x1 = c_x - w / 2
                new_y1 = c_y - h / 2
                new_x2 = c_x + w / 2
                new_y2 = c_y + h / 2

                # 이미지 경계를 넘지 않도록 조정
                if new_x1 < 0:
                    new_x2 -= new_x1  # 좌측 초과분 만큼 우측으로 이동
                    new_x1 = 0
                if new_y1 < 0:
                    new_y2 -= new_y1
                    new_y1 = 0
                if new_x2 > self.img_w:
                    overflow = new_x2 - self.img_w
                    new_x1 -= overflow
                    new_x2 = self.img_w
                if new_y2 > self.img_h:
                    overflow = new_y2 - self.img_h
                    new_y1 -= overflow
                    new_y2 = self.img_h

                # 다시 정렬 (혹시라도 x1 > x2 되는 경우 방지)
                new_x1 = max(0, new_x1)
                new_y1 = max(0, new_y1)
                new_x2 = min(self.img_w, new_x2)
                new_y2 = min(self.img_h, new_y2)

                new_box = [new_x1, new_y1, new_x2, new_y2]
                new_bboxes.append(new_box)

            boxes = np.stack(new_bboxes)
            right = self.right

        return boxes, right

        # self.bbox_cnt += 1
        # if len(self.prev_uvd) < 1 or self.bbox_cnt > self.det_cooltime:
        #     results = self.detector.predict(img, conf=conf, verbose=verbose, max_det=5)[0]
        #     self.bbox_cnt = 0
        # else:


        ## new hand detection. need to add re-initialization

        # bbox = self.default_bbox
        # self.bbox_cnt += 1
        #
        # if self.bbox_cnt > self.det_cooltime:
        #     if self.prev_coord is not None:
        #         self.bbox_togo = self.calc_bbox_coords(image_width, image_height, self.prev_coord)
        #         self.bbox_cnt = 0
        #
        # if self.bbox_togo is not None:
        #     for idx in range(2):
        #         if np.abs(bbox[idx] - self.bbox_togo[idx]) > 1:
        #             if bbox[idx] > self.bbox_togo[idx]:
        #                 bbox[idx] -= self.pixelperframe
        #             else:
        #                 bbox[idx] += self.pixelperframe
        #         else:
        #             bbox[idx] = self.bbox_togo[idx]
        #
        #     self.default_bbox = bbox
        #     if bbox[0:2] == self.bbox_togo[0:2]:
        #         self.bbox_togo = None

        ##


def calc_pred_center(coords):
    x_min = np.min(coords[:, 0])
    y_min = np.min(coords[:, 1])
    x_max = np.max(coords[:, 0])
    y_max = np.max(coords[:, 1])

    x_c = (x_min + x_max) / 2
    y_c = (y_min + y_max) / 2

    return x_c, y_c


def project_full_img(points, cam_trans, focal_length, img_res):
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = camera_center[0]
    K[1, 2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:]

    V_2d = (K @ points.T).T
    return V_2d[..., :-1]

def log_event(label, flag_time=False):
    global prev_label, prev, log_t_dict

    now = time.time()
    latency = now - prev
    if flag_time:
        print(f"{prev_label} ~ {label}: {latency:.4f}")
    prev_label = label
    prev = now

    # if label in log_t_dict:
    #     log_t_dict[label].append(latency)
        # if len(log_t_dict[label]) > 100:
        #     print(f"avg latency for {label}: {np.average(np.array(log_t_dict[label]))}")



def main():
    from collections import deque

    torch.backends.cudnn.benchmark = True
    tracker = HandTracker_wilor()

    if flag_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            exit()


    queue_righthand = deque([], maxlen=10)
    img_idx = 0
    flag_gesture = False
    try:
        while True:
            img_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            # input
            if flag_webcam:
                ret, color_image = cap.read()
                if not ret:
                    print("이미지를 받아올 수 없습니다.")
                    break

                # webcam : 480 640 3
                color_image = color_image[60:-60, :, :]

            elif flag_rsrecord: # realsense : 480 640
                frames = pipeline.wait_for_frames()

                aligned_frames = align.process(frames)

                # 정렬된 depth와 color 프레임 추출
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                depth_image_colored = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # color_image = cv2.resize(color_image, dsize=(tracker.img_w, tracker.img_h), interpolation=cv2.INTER_CUBIC)
                # depth_image_colored = cv2.resize(depth_image_colored, dsize=(tracker.img_w, tracker.img_h),
                #                          interpolation=cv2.INTER_CUBIC)


                depth_image = np.asanyarray(depth_frame.get_data())  # 이건 uint16 형식
                # 깊이 값을 float(m 단위)로 변환하려면:
                depth_image_float = depth_image * depth_scale  # float32 형식으로 변환

                color_image = color_image[60:-60, :, :]
                depth_image_colored = depth_image_colored[60:-60, :, :]
                depth_image_float = depth_image_float[60:-60, :]        # m scale

                # cv2.imshow("RGB from .bag", color_image)
                # cv2.imshow("Depth from .bag", depth_image_colored)

            # cv2.imshow("input", color_image)

            outs = tracker.run(color_image, flag_time)
            if not outs:
                continue

            all_right, all_uvds, all_verts, all_cam_t = outs

            ### Render mesh image
            if flag_render_mesh:
                misc_args = dict(
                    mesh_base_color=LIGHT_PURPLE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=tracker.scaled_focal_length,
                )
                cam_view = tracker.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=[tracker.img_w, tracker.img_h],
                                                         is_right=all_right, **misc_args)

                # Overlay image
                input_img = color_image.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
                input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

                output_img = input_img_overlay[:, :, ::-1]
                # cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}.jpg'), 255 * output_img)
                # cv2.imwrite(os.path.join(args.out_folder, f'{img_idx}.jpg'), 255 * output_img)
                cv2.imshow("mesh result", output_img)

            ### Draw skeleton
            output_img_skel = color_image.copy()
            for uvd in all_uvds:
                output_img_skel = draw_2d_skeleton(output_img_skel, uvd)
            cv2.imshow("skeleton result", output_img_skel)

            log_event("render results", flag_time)

            print(" ------------------------------------ ")

            ## nearby object detection ##
            """
            - 손 위치 depth로부터 10cm 내에서 detect된 물체 boundingbox만 체크
            - palm 2D pose로부터 10cm 내에 물체 bb가 있으면 gesture recognizer 활성화. 
            """
            log_event("start contact detect",  flag_time=True)

            indices = np.where(np.asarray(all_right) == 0)[0]  ### check. 0: left, 1: right

            if len(indices) > 0:
                uvd = np.squeeze(np.asarray(all_uvds)[indices[0]])
                uv_wrist = uvd[0, :-1]
                d_wrist = depth_image_float[int(uv_wrist[1]), int(uv_wrist[0])]

                palm_points_2d = uvd[[0, 4, 8, 12, 16, 20], :2]
                palm_uv = np.mean(palm_points_2d, axis=0)

                obj_nearby_list = tracker.detect_objs(color_image, depth_image_float, d_wrist)

                # 손 근처 물체 BB 시각화.
                if len(obj_nearby_list) > 0:
                    vis_obj = color_image.copy()
                    for obj_nearby in obj_nearby_list:
                        # draw every nearby object bb in green
                        cv2.rectangle(vis_obj, (obj_nearby[0], obj_nearby[1]), (obj_nearby[2], obj_nearby[3]), (0, 255, 0), 2)
                        cv2.putText(vis_obj, f'{obj_nearby[-1]}', (obj_nearby[0], obj_nearby[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    ## activate gesture recognizer ##
                    for obj_nearby in obj_nearby_list:
                        cx, cy = (obj_nearby[0]+obj_nearby[2])//2, (obj_nearby[1]+obj_nearby[3])//2

                        distance = np.sqrt((cx - palm_uv[0]) ** 2 + (cy - palm_uv[1]) ** 2)
                        print("distance : ", distance)
                        if distance < 50.0:
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

                elif tracker.flag_detected:     # off the gesture only when no nearby bb has found from obj detection
                    flag_gesture = False

            log_event("end contact detect", flag_time=True)

            print("Activate gesture :    ", flag_gesture)



    finally:
        if flag_webcam:
            cap.release()
        elif flag_rsrecord:
            pipeline.stop()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()



