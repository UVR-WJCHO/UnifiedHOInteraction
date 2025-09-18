import copy
import os
# os.environ["PYOPENGL_PLATFORM"] = "OSMesa"
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))     # append current dir to PATH
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../"))

import torch
from tqdm import tqdm
import cv2
import time
import torchvision.transforms as standard
import numpy as np

from base import Tester
from config import cfg
from utils.visualize import draw_2d_skeleton
from data.processing import inference_extraHM, augmentation, cv2pil, augmentation_real
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


prev = time.time()
prev_label = "init"

log_t_dict = {"start": [], "detection": [], "preprocess":[], "inference":[],"postprocess":[]}


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
    #
    #     if len(log_t_dict[label]) > 100:
    #         print(f"avg latency for {label}: {np.average(np.array(log_t_dict[label]))}")



def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def encode_hand_data(hand_result):
    """ Encode hand data to json format """

    """ Example """
    """
    handDataPackage['joints_0']
    handDataPackage['joints_1']
    if the hand is not detected, returns zero value joints 

    currently consider single hand
    """
    handDataPackage = dict()
    joints = list()
    num_joints = 21

    for joint_uvd in hand_result:
        for id in range(num_joints):
            joint = dict()
            joint['id'] = int(id)
            joint['u'] = float(joint_uvd[id, 0])
            joint['v'] = float(joint_uvd[id, 1])
            joint['d'] = float(joint_uvd[id, 2])
            joints.append(joint)
        break
    handDataPackage['joints'] = joints

    return handDataPackage


class HandTracker():
    def __init__(self):
        self.tester = Tester()
        self.tester._make_model()
        # self.detector = HandDetector()

        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = standard.Compose([standard.ToTensor(), standard.Normalize(*mean_std)])

        if cfg.extra:
            self.extra_uvd_left = np.zeros((21, 3), dtype=np.float32)
            self.extra_uvd_right = np.zeros((21, 3), dtype=np.float32)
            self.idx = 0


        # self.prev_bbox_list = []
        # self.prev_flag_flip_list = []
        self.mediahand = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

        # do first iteration
        emptyImg = np.ones((256, 256), dtype=float)
        img_pil = cv2pil(emptyImg)
        img = self.transform(img_pil)
        img = torch.unsqueeze(img, 0).type(torch.float32)
        inputs = {'img': img}

        self.extra_uvd = np.zeros((21, 3), dtype=np.float32)
        extra_hm = inference_extraHM(self.extra_uvd, self.idx, reinit_num=10)
        inputs['extra'] = torch.unsqueeze(torch.from_numpy(extra_hm), dim=0)
        _ = self.tester.model(inputs)
        print("success on first run")

        self.img_w = 640
        self.img_h = 360
        self.crop_size = 360
        self.default_bbox = [240, 0, self.crop_size, self.crop_size]

        self.prev_coord = None

        self.winname = "our result"
        # cv2.namedWindow(self.winname)
        # cv2.moveWindow(self.winname, 2640, 400)

        ## update target bbox for every 10 frame, move current bbox to target frame for 2 pixel per frame.
        self.bbox_cooltime = 10
        self.bbox_cnt = 0
        self.bbox_togo = None
        self.pixelperframe = 2


    def Process_single_newroi(self, img): # input : img_cv
        log_event("start")
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
        image_height, image_width = img.shape[0], img.shape[1]  # (360, 640)

        # img_cv = np.copy(img)

        ## new hand detection. need to add re-initialization

        bbox = self.default_bbox
        self.bbox_cnt += 1

        if self.bbox_cnt > self.bbox_cooltime:
            if self.prev_coord is not None:
                self.bbox_togo = self.calc_bbox_coords(image_width, image_height, self.prev_coord)
                self.bbox_cnt = 0

        if self.bbox_togo is not None:
            for idx in range(2):
                if np.abs(bbox[idx] - self.bbox_togo[idx]) > 1:
                    if bbox[idx] > self.bbox_togo[idx]:
                        bbox[idx] -= self.pixelperframe
                    else:
                        bbox[idx] += self.pixelperframe
                else:
                    bbox[idx] = self.bbox_togo[idx]

            self.default_bbox = bbox
            if bbox[0:2] == self.bbox_togo[0:2]:
                self.bbox_togo = None

        log_event("detection")

        img_crop, img2bb_trans, bb2img_trans, _, _, = augmentation_real(img, bbox, flip=False)
        # cv2.imshow('img_crop', img_crop/255.0)
        # cv2.waitKey(1)


        # transform img
        img_pil = cv2pil(img_crop)
        img = self.transform(img_pil)
        img = torch.unsqueeze(img, 0).type(torch.float32)
        inputs = {'img': img}


        if cfg.extra:
            # self.extra_uvd = np.copy(self.extra_uvd_right)
            self.extra_uvd = np.zeros((21, 3), dtype=np.float32)

            # affine transform x,y coordinates with current crop info
            uv1 = np.concatenate((self.extra_uvd[:, :2], np.ones_like(self.extra_uvd[:, :1])), 1)
            self.extra_uvd[:, :2] = np.dot(img2bb_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]

            # normalize uv, depth is already relative value
            self.extra_uvd[:, :2] = self.extra_uvd[:, :2] / (cfg.input_img_shape[0] // 2) - 1

            extra_hm = inference_extraHM(self.extra_uvd, self.idx, reinit_num=10)
            inputs['extra'] = torch.unsqueeze(torch.from_numpy(extra_hm), dim=0)
            self.idx += 1

        log_event("preprocess")
        with torch.no_grad():
            outs = self.tester.model(inputs).detach()
        log_event("inference")

        outs = outs.to("cpu", non_blocking=True)
        coords_uvd = outs.numpy()[0]

        # normalized value to uv(pixel) range
        coords_uvd[:, :2] = (coords_uvd[:, :2] + 1) * (cfg.input_img_shape[0] // 2)

        # back to original image
        uv1 = np.concatenate((coords_uvd[:, :2], np.ones_like(coords_uvd[:, :1])), 1)
        coords_uvd[:, :2] = np.dot(bb2img_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]


        if cfg.extra:
            self.extra_uvd_right = np.copy(coords_uvd[cfg.num_vert:])

        # restore depth value after passing extra pose
        coords_uvd[:, 2] = coords_uvd[:, 2] * cfg.depth_box # + root_depth (we don't know here)

        # mesh_uvd = copy.deepcopy(all_uvd[:cfg.num_vert])  # (778, 3)
        coords_uvd = coords_uvd[cfg.num_vert:]   # (21, 3)

        t3 = time.time()
        # print("preprocess, inference, postprocess : ", t1-t0, t2-t1, t3-t2)

        ### visualize output in server ###
        # img_cv = draw_2d_skeleton(img_cv, coords_uvd)
        # cv2.imshow(self.winname, img_cv)
        # cv2.waitKey(1)

        self.prev_coord = np.copy(coords_uvd)

        log_event("postprocess")

        return coords_uvd





    def Process_single_nomp(self, img): # input : img_cv
        t0 = time.time()
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
        imgSize = (img.shape[0], img.shape[1])  # (360, 640)

        # image from hololens2 is fliped both direction
        # img = cv2.flip(img, 0)
        img_cv = np.copy(img)

        ## new hand detection. need to add re-initialization
        if self.prev_coord is not None:
            bbox = self.calc_bounding_rect_coords(self.img_w, self.img_h, self.prev_coord)
        else:
            bbox = self.default_bbox


        img_crop, img2bb_trans, bb2img_trans, _, _, = augmentation_real(img, bbox, flip=False)
        cv2.imshow('img_crop', img_crop/255.0)
        cv2.waitKey(1)


        # transform img
        img_pil = cv2pil(img_crop)
        img = self.transform(img_pil)
        img = torch.unsqueeze(img, 0).type(torch.float32)
        inputs = {'img': img}


        if cfg.extra:
            self.extra_uvd = np.copy(self.extra_uvd_right)

            # affine transform x,y coordinates with current crop info
            uv1 = np.concatenate((self.extra_uvd[:, :2], np.ones_like(self.extra_uvd[:, :1])), 1)
            self.extra_uvd[:, :2] = np.dot(img2bb_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]

            # normalize uv, depth is already relative value
            self.extra_uvd[:, :2] = self.extra_uvd[:, :2] / (cfg.input_img_shape[0] // 2) - 1

            extra_hm = inference_extraHM(self.extra_uvd, self.idx, reinit_num=10)
            inputs['extra'] = torch.unsqueeze(torch.from_numpy(extra_hm), dim=0)
            self.idx += 1

        t1 = time.time()
        with torch.no_grad():
            outs = self.tester.model(inputs).detach()
        t2 = time.time()

        outs = outs.to("cpu", non_blocking=True)
        coords_uvd = outs.numpy()[0]

        # normalized value to uv(pixel) range
        coords_uvd[:, :2] = (coords_uvd[:, :2] + 1) * (cfg.input_img_shape[0] // 2)

        # back to original image
        uv1 = np.concatenate((coords_uvd[:, :2], np.ones_like(coords_uvd[:, :1])), 1)
        coords_uvd[:, :2] = np.dot(bb2img_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]


        if cfg.extra:
            self.extra_uvd_right = np.copy(coords_uvd[cfg.num_vert:])

        # restore depth value after passing extra pose
        coords_uvd[:, 2] = coords_uvd[:, 2] * cfg.depth_box # + root_depth (we don't know here)

        # mesh_uvd = copy.deepcopy(all_uvd[:cfg.num_vert])  # (778, 3)
        coords_uvd = coords_uvd[cfg.num_vert:]   # (21, 3)

        t3 = time.time()
        # print("preprocess, inference, postprocess : ", t1-t0, t2-t1, t3-t2)

        ### visualize output in server ###
        img_cv = draw_2d_skeleton(img_cv, coords_uvd)
        cv2.imshow(self.winname, img_cv)
        cv2.waitKey(1)

        return coords_uvd


    def Process_single(self, img): # input : img_cv
        t0 = time.time()
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
        imgSize = (img.shape[0], img.shape[1])  # (360, 640)

        # image from hololens2 is fliped both direction
        img = cv2.flip(img, 0)
        img_cv = np.copy(img)

        ### hand detection with mediapipe (17ms)
        # currently extracting only right-side hand
        bbox_list, img_crop_list, img2bb_trans_list, bb2img_trans_list, flag_flip_list = self.extract_singlehand(img, imgSize)




        t1 = time.time()
        joint_uvd_list = []
        # mesh_uvd_list = []

        if len(bbox_list) == 0:
            print("no bbox, return zero joint")
            joint_uvd = np.zeros((21, 3), dtype=np.float32)
            joint_uvd_list.append(joint_uvd)
            return joint_uvd_list

        else:
            for bbox, img_crop, img2bb_trans, bb2img_trans, flag_flip in \
                    zip(bbox_list, img_crop_list, img2bb_trans_list, bb2img_trans_list, flag_flip_list):
                # transform img
                img_pil = cv2pil(img_crop)
                img = self.transform(img_pil)
                img = torch.unsqueeze(img, 0).type(torch.float32)
                inputs = {'img': img}

                if cfg.extra:
                    if flag_flip:
                        self.extra_uvd = np.copy(self.extra_uvd_left)
                    else:
                        self.extra_uvd = np.copy(self.extra_uvd_right)

                    # affine transform x,y coordinates with current crop info
                    uv1 = np.concatenate((self.extra_uvd[:, :2], np.ones_like(self.extra_uvd[:, :1])), 1)
                    self.extra_uvd[:, :2] = np.dot(img2bb_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]

                    # normalize uv, depth is already relative value
                    self.extra_uvd[:, :2] = self.extra_uvd[:, :2] / (cfg.input_img_shape[0] // 2) - 1

                    extra_hm = inference_extraHM(self.extra_uvd, self.idx, reinit_num=10)
                    inputs['extra'] = torch.unsqueeze(torch.from_numpy(extra_hm), dim=0)
                    self.idx += 1

                t2 = time.time()
                with torch.no_grad():
                    outs = self.tester.model(inputs).detach()
                t3 = time.time()

                outs = outs.to("cpu", non_blocking=True)
                coords_uvd = outs.numpy()[0]


                # normalized value to uv(pixel) range
                coords_uvd[:, :2] = (coords_uvd[:, :2] + 1) * (cfg.input_img_shape[0] // 2)

                # back to original image
                uv1 = np.concatenate((coords_uvd[:, :2], np.ones_like(coords_uvd[:, :1])), 1)
                coords_uvd[:, :2] = np.dot(bb2img_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]
                t5 = time.time()

                if cfg.extra:
                    if flag_flip:
                        self.extra_uvd_left = np.copy(coords_uvd[cfg.num_vert:])
                    else:
                        self.extra_uvd_right = np.copy(coords_uvd[cfg.num_vert:])

                # restore depth value after passing extra pose
                coords_uvd[:, 2] = coords_uvd[:, 2] * cfg.depth_box # + root_depth (we don't know)

                # mesh_uvd = copy.deepcopy(all_uvd[:cfg.num_vert])  # (778, 3)
                coords_uvd = coords_uvd[cfg.num_vert:]   # (21, 3)
                if flag_flip:
                    coords_uvd[:, 0] = imgSize[1] - coords_uvd[:, 0]
                    # mesh_uvd[:, 0] = imgSize[1] - mesh_uvd[:, 0]

                joint_uvd_list.append(coords_uvd)
                # mesh_uvd_list.append(mesh_uvd)

                t4 = time.time()
                print("detect, preprocess, inference, postprocess : ", t1-t0, t2-t1, t3-t2, t4-t3)

            ### visualize output in server ###
            # for joint_uvd in joint_uvd_list:
            #     img_cv = draw_2d_skeleton(img_cv, joint_uvd)
            # cv2.imshow('img_cv', img_cv)
            # cv2.waitKey(1)

            return joint_uvd_list


    def calc_bbox_coords(self, image_width, image_height, coords):
        x_min = np.min(coords[:, 0])
        y_min = np.min(coords[:, 1])
        x_max = np.max(coords[:, 0])
        y_max = np.max(coords[:, 1])

        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2

        x_min = max(0, x_c - self.crop_size / 2)  # *3)
        y_min = max(0, y_c - self.crop_size / 2)

        if (x_min + self.crop_size) > image_width:
            x_min = image_width - self.crop_size
        if (y_min + self.crop_size) > image_height:
            y_min = image_height - self.crop_size

        bbox = [x_min, y_min, self.crop_size, self.crop_size]

        return bbox


    def calc_bounding_rect_coords(self, image_width, image_height, coords):
        x, y, w, h = cv2.boundingRect(coords[:, :2])
        # x, y : upper right point

        x = image_width - x
        x_c = x + w/2
        y_c = y + h/2

        x_min = max(0, x_c - self.crop_size/2)  # *3)
        y_min = max(0, y_c - self.crop_size/2)

        if (x_min + self.crop_size) > image_width:
            x_min = image_width - self.crop_size
        if (y_min + self.crop_size) > image_height:
            y_min = image_height - self.crop_size

        bbox = [x_min, y_min, self.crop_size, self.crop_size]

        return bbox

    def create_input(self, img, bbox_list, width):
        img_crop_list, img2bb_trans_list, bb2img_trans_list = [], [], []

        if len(bbox_list) == 1:
            if bbox_list[0][0] < (width / 2):
                flag_flip_list = [True]
            else:
                flag_flip_list = [False]

            img_crop, img2bb_trans, bb2img_trans, _, _, = augmentation_real(img, bbox_list[0], flip=flag_flip_list[0])
            # cv2.imshow('crop 0', img_crop / 255.)
            # cv2.waitKey(1)
            img_crop_list.append(img_crop)
            img2bb_trans_list.append(img2bb_trans)
            bb2img_trans_list.append(bb2img_trans)

        else:
            if bbox_list[0][0] < bbox_list[1][0]:
                flag_flip_list = [True, False]      # hand order : left - right
            else:
                flag_flip_list = [False, True]      # hand order : right - left

            for idx, bbox in enumerate(bbox_list):
                flag_flip = flag_flip_list[idx]
                img_crop, img2bb_trans, bb2img_trans, _, _, = augmentation_real(img, bbox, flip=flag_flip)
                # cv2.imshow('crop 0', img_crop / 255.)
                # cv2.waitKey(1)
                img_crop_list.append(img_crop)
                img2bb_trans_list.append(img2bb_trans)
                bb2img_trans_list.append(bb2img_trans)

        return bbox_list, img_crop_list, img2bb_trans_list, bb2img_trans_list, flag_flip_list


def main():
    torch.backends.cudnn.benchmark = True
    tracker = HandTracker()
    cam_intrinsic = None

    for i in range(10):
        color = np.random.randint(255, size=(640, 480, 3), dtype=np.uint8)
        pred_list = tracker.Process(color)
        ### if required uvd format ##

        for all_uvd in pred_list:
            mesh_uvd = copy.deepcopy(all_uvd[:cfg.num_vert])       # (778, 3)
            joint_uvd = copy.deepcopy(all_uvd[cfg.num_vert:])       # (21, 3)
            ### if required xyz format ###
            # all_xyz = uvd2xyz(all_uvd, cam_intrinsic)
            # _visualize(color, joint_uvd)
            print("joint : ", joint_uvd)

    print("test end")



def _get_input(frame):
    ### load image from recorded files ###
    load_filepath = './recorded_files/'

    color = cv2.imread(load_filepath + 'color_%d.png' % frame)
    color = cv2.resize(color, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    return color

def _visualize(color, coords_uvd):
    vis = draw_2d_skeleton(color, coords_uvd[cfg.num_vert:])
    vis = cv2.resize(vis, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)
    color = cv2.resize(color, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("vis", vis)
    cv2.imshow("img", color)
    cv2.waitKey(50)

def uvd2xyz(uvd, K):
    fx, fy, fu, fv = K[0, 0], K[0, 0], K[0, 2], K[1, 2]
    xyz = np.zeros_like(uvd, np.float32)
    xyz[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
    xyz[:, 1] = (uvd[:, 1] - fv) * uvd[:, 2] / fy
    xyz[:, 2] = uvd[:, 2]
    return xyz

def xyz2uvd(xyz, K):
    fx, fy, fu, fv = K[0, 0], K[0, 0], K[0, 2], K[1, 2]
    uvd = np.zeros_like(xyz, np.float32)
    uvd[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
    uvd[:, 2] = xyz[:, 2]
    return uvd

if __name__ == '__main__':
    main()



