import os
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import json
import logging
script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import load_torch_file, ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

from .models.onnx_models import ViTPose, Yolo
from .pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, crop, bbox_from_detector
from .utils import get_face_bboxes, padding_resize, resize_by_area, resize_to_bounds
from .pose_utils.human_visualization import AAPoseMeta, draw_aapose_by_meta_new, draw_aaface_by_meta
from .retarget_pose import get_retarget_pose

class OnnxDetectionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": (folder_paths.get_filename_list("detection"), {"tooltip": "These models are loaded from the 'ComfyUI/models/detection' -folder",}),
                "yolo_model": (folder_paths.get_filename_list("detection"), {"tooltip": "These models are loaded from the 'ComfyUI/models/detection' -folder",}),
                "onnx_device": (["CUDAExecutionProvider", "CPUExecutionProvider"], {"default": "CUDAExecutionProvider", "tooltip": "Device to run the ONNX models on"}),
            },
        }

    RETURN_TYPES = ("POSEMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Loads ONNX models for pose and face detection. ViTPose for pose estimation and YOLO for object detection."

    def loadmodel(self, vitpose_model, yolo_model, onnx_device):

        vitpose_model_path = folder_paths.get_full_path_or_raise("detection", vitpose_model)
        yolo_model_path = folder_paths.get_full_path_or_raise("detection", yolo_model)

        vitpose = ViTPose(vitpose_model_path, onnx_device)
        yolo = Yolo(yolo_model_path, onnx_device)

        model = {
            "vitpose": vitpose,
            "yolo": yolo,
        }

        return (model, )

class PoseAndFaceDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Height of the generation"}),
            },
            "optional": {
                "retarget_image": ("IMAGE", {"default": None, "tooltip": "Optional reference image for pose retargeting"}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "STRING", "BBOX", "BBOX,")
    RETURN_NAMES = ("pose_data", "face_images", "key_frame_body_points", "bboxes", "face_bboxes")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Detects human poses and face images from input images. Optionally retargets poses based on a reference image."

    def process(self, model, images, width, height, retarget_image=None):
        detector = model["yolo"]
        pose_model = model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()
        if retarget_image is not None:
            refer_img = resize_by_area(retarget_image[0].numpy() * 255, width * height, divisor=16) / 255.0
            ref_bbox = (detector(
                cv2.resize(refer_img.astype(np.float32), (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])

            if ref_bbox is None or ref_bbox[-1] <= 0 or (ref_bbox[2] - ref_bbox[0]) < 10 or (ref_bbox[3] - ref_bbox[1]) < 10:
                ref_bbox = np.array([0, 0, refer_img.shape[1], refer_img.shape[0]])

            center, scale = bbox_from_detector(ref_bbox, input_resolution, rescale=rescale)
            refer_img = crop(refer_img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (refer_img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            ref_keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            refer_pose_meta = load_pose_metas_from_kp2ds_seq(ref_keypoints, width=retarget_image.shape[2], height=retarget_image.shape[1])[0]

        comfy_pbar = ProgressBar(B*2)
        progress = 0
        bboxes = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting bboxes"):
            bboxes.append(detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        kp2ds = []
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting keypoints"):
            if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])

            bbox_xywh = bbox
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        kp2ds = np.concatenate(kp2ds, 0)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)

        face_images = []
        face_bboxes = []
        for idx, meta in enumerate(pose_metas):
            face_bbox_for_image = get_face_bboxes(meta['keypoints_face'][:, :2], scale=1.3, image_shape=(H, W))
            x1, x2, y1, y2 = face_bbox_for_image
            face_bboxes.append((x1, y1, x2, y2))
            face_image = images_np[idx][y1:y2, x1:x2]
            # Check if face_image is valid before resizing
            if face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
                logging.warning(f"Empty face crop on frame {idx}, creating fallback image.")
                # Create a fallback image (black or use center crop)
                fallback_size = int(min(H, W) * 0.3)
                fallback_x1 = (W - fallback_size) // 2
                fallback_x2 = fallback_x1 + fallback_size
                fallback_y1 = int(H * 0.1)
                fallback_y2 = fallback_y1 + fallback_size
                face_image = images_np[idx][fallback_y1:fallback_y2, fallback_x1:fallback_x2]
                
                # If still empty, create a black image
                if face_image.size == 0:
                    face_image = np.zeros((fallback_size, fallback_size, C), dtype=images_np.dtype)
            face_image = cv2.resize(face_image, (512, 512))
            face_images.append(face_image)

        face_images_np = np.stack(face_images, 0)
        face_images_tensor = torch.from_numpy(face_images_np)

        if retarget_image is not None and refer_pose_meta is not None:
            retarget_pose_metas = get_retarget_pose(pose_metas[0], refer_pose_meta, pose_metas, None, None)
        else:
            retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas]

        bbox = np.array(bboxes[0]).flatten()
        if bbox.shape[0] >= 4:
            bbox_ints = tuple(int(v) for v in bbox[:4])
        else:
            bbox_ints = (0, 0, 0, 0)

        key_frame_num = 4 if B >= 4 else 1
        key_frame_step = len(pose_metas) // key_frame_num
        key_frame_index_list = list(range(0, len(pose_metas), key_frame_step))

        key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]

        for key_frame_index in key_frame_index_list:
            keypoints_body_list = []
            body_key_points = pose_metas[key_frame_index]['keypoints_body']
            for each_index in key_points_index:
                each_keypoint = body_key_points[each_index]
                if None is each_keypoint:
                    continue
                keypoints_body_list.append(each_keypoint)

            keypoints_body = np.array(keypoints_body_list)[:, :2]
            wh = np.array([[pose_metas[0]['width'], pose_metas[0]['height']]])
            points = (keypoints_body * wh).astype(np.int32)
            points_dict_list = []
            for point in points:
                points_dict_list.append({"x": int(point[0]), "y": int(point[1])})

        pose_data = {
            "retarget_image": refer_img if retarget_image is not None else None,
            "pose_metas": retarget_pose_metas,
            "refer_pose_meta": refer_pose_meta if retarget_image is not None else None,
            "pose_metas_original": pose_metas,
        }

        return (pose_data, face_images_tensor, json.dumps(points_dict_list), [bbox_ints], face_bboxes)

class DrawViTPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Height of the generation"}),
                "retarget_padding": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1, "tooltip": "When > 0, the retargeted pose image is padded and resized to the target size"}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks. Set to 0 to disable body drawing, -1 for auto"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the hand sticks. Set to 0 to disable hand drawing, -1 for auto"}),
                "draw_head": ("BOOLEAN", {"default": "True", "tooltip": "Whether to draw head keypoints"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("pose_images", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Draws pose images from pose data."

    def process(self, pose_data, width, height, body_stick_width, hand_stick_width, draw_head, retarget_padding=64):

        retarget_image = pose_data.get("retarget_image", None)
        pose_metas = pose_data["pose_metas"]

        draw_hand = hand_stick_width != 0
        use_retarget_resize = retarget_padding > 0 and retarget_image is not None

        comfy_pbar = ProgressBar(len(pose_metas))
        progress = 0
        crop_target_image = None
        pose_images = []

        for idx, meta in enumerate(tqdm(pose_metas, desc="Drawing pose images")):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            pose_image = draw_aapose_by_meta_new(canvas, meta, draw_hand=draw_hand, draw_head=draw_head, body_stick_width=body_stick_width, hand_stick_width=hand_stick_width)

            if crop_target_image is None:
                crop_target_image = pose_image

            if use_retarget_resize:
                pose_image = resize_to_bounds(pose_image, height, width, crop_target_image=crop_target_image, extra_padding=retarget_padding)
            else:
                pose_image = padding_resize(pose_image, height, width)

            pose_images.append(pose_image)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_images_np = np.stack(pose_images, 0)
        pose_images_tensor = torch.from_numpy(pose_images_np).float() / 255.0

        return (pose_images_tensor, )

class PoseRetargetPromptHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", )
    RETURN_NAMES = ("prompt", "retarget_prompt", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Generates text prompts for pose retargeting based on visibility of arms and legs in the template pose. Originally used for Flux Kontext"

    def process(self, pose_data):
        refer_pose_meta = pose_data.get("refer_pose_meta", None)
        if refer_pose_meta is None:
            return ("Change the person to face forward.", "Change the person to face forward.", )
        tpl_pose_metas = pose_data["pose_metas_original"]
        arm_visible = False
        leg_visible = False

        for tpl_pose_meta in tpl_pose_metas:
            tpl_keypoints = tpl_pose_meta['keypoints_body']
            tpl_keypoints = np.array(tpl_keypoints)
            if np.any(tpl_keypoints[3]) != 0 or np.any(tpl_keypoints[4]) != 0 or np.any(tpl_keypoints[6]) != 0 or np.any(tpl_keypoints[7]) != 0:
                if (tpl_keypoints[3][0] <= 1 and tpl_keypoints[3][1] <= 1 and tpl_keypoints[3][2] >= 0.75) or (tpl_keypoints[4][0] <= 1 and tpl_keypoints[4][1] <= 1 and tpl_keypoints[4][2] >= 0.75) or \
                    (tpl_keypoints[6][0] <= 1 and tpl_keypoints[6][1] <= 1 and tpl_keypoints[6][2] >= 0.75) or (tpl_keypoints[7][0] <= 1 and tpl_keypoints[7][1] <= 1 and tpl_keypoints[7][2] >= 0.75):
                    arm_visible = True
            if np.any(tpl_keypoints[9]) != 0 or np.any(tpl_keypoints[12]) != 0 or np.any(tpl_keypoints[10]) != 0 or np.any(tpl_keypoints[13]) != 0:
                if (tpl_keypoints[9][0] <= 1 and tpl_keypoints[9][1] <= 1 and tpl_keypoints[9][2] >= 0.75) or (tpl_keypoints[12][0] <= 1 and tpl_keypoints[12][1] <= 1 and tpl_keypoints[12][2] >= 0.75) or \
                    (tpl_keypoints[10][0] <= 1 and tpl_keypoints[10][1] <= 1 and tpl_keypoints[10][2] >= 0.75) or (tpl_keypoints[13][0] <= 1 and tpl_keypoints[13][1] <= 1 and tpl_keypoints[13][2] >= 0.75):
                    leg_visible = True
            if arm_visible and leg_visible:
                break

        if leg_visible:
            if tpl_pose_meta['width'] > tpl_pose_meta['height']:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."

            if refer_pose_meta['width'] > refer_pose_meta['height']:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."
        elif arm_visible:
            if tpl_pose_meta['width'] > tpl_pose_meta['height']:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."

            if refer_pose_meta['width'] > refer_pose_meta['height']:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."
        else:
            tpl_prompt = "Change the person to face forward."
            refer_prompt = "Change the person to face forward."

        return (tpl_prompt, refer_prompt, )

class PoseBoneManipulation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "head_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "tooltip": "Scale factor for head size"}),
                "head_offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Horizontal offset for head (normalized)"}),
                "head_offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Vertical offset for head (normalized)"}),
                "neck_length_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "tooltip": "Scale factor for neck length (longer/shorter neck)"}),
                "neck_offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Horizontal offset for neck (head tilt/lean)"}),
                "torso_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "tooltip": "Scale factor for torso height"}),
                "torso_width_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "tooltip": "Scale factor for torso width (shoulders and hips)"}),
                "torso_offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Horizontal offset for torso"}),
                "torso_offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Vertical offset for torso"}),
                "shoulders_width_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "tooltip": "Scale factor for shoulder width"}),
                "shoulders_offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Vertical offset for shoulders"}),
                "left_arm_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "tooltip": "Scale factor for left arm length"}),
                "left_arm_offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Horizontal offset for left arm"}),
                "left_arm_offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Vertical offset for left arm"}),
                "right_arm_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "tooltip": "Scale factor for right arm length"}),
                "right_arm_offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Horizontal offset for right arm"}),
                "right_arm_offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Vertical offset for right arm"}),
                "left_leg_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "tooltip": "Scale factor for left leg length"}),
                "left_leg_offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Horizontal offset for left leg"}),
                "left_leg_offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Vertical offset for left leg"}),
                "right_leg_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "tooltip": "Scale factor for right leg length"}),
                "right_leg_offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Horizontal offset for right leg"}),
                "right_leg_offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Vertical offset for right leg"}),
                "left_hand_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "tooltip": "Scale factor for left hand size"}),
                "left_hand_offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Horizontal offset for left hand"}),
                "left_hand_offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Vertical offset for left hand"}),
                "right_hand_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "tooltip": "Scale factor for right hand size"}),
                "right_hand_offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Horizontal offset for right hand"}),
                "right_hand_offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Vertical offset for right hand"}),
            },
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Manipulate individual bones in the pose skeleton by scaling and offsetting them for cartoonish effects."

    def manipulate_keypoints(self, kps, indices, anchor_idx, scale, offset_x, offset_y, ref_width, ref_height):
        """
        Manipulate a group of keypoints by scaling and offsetting.

        Args:
            kps: keypoint array (x, y coordinates)
            indices: list of keypoint indices to manipulate
            anchor_idx: index of the anchor point (or None to use center)
            scale: scale factor
            offset_x, offset_y: offset in normalized coordinates (0-1)
            ref_width, ref_height: reference dimensions for offset calculation (skeleton or image dimensions)
        """
        if kps is None or len(indices) == 0:
            return

        # Get anchor point
        if anchor_idx is not None and anchor_idx < len(kps):
            anchor = kps[anchor_idx].copy()
        else:
            # Use center of the keypoints as anchor
            anchor = np.mean(kps[indices], axis=0)

        # Scale keypoints relative to anchor
        if scale != 1.0:
            for idx in indices:
                if idx < len(kps):
                    # Calculate vector from anchor to keypoint
                    vec = kps[idx] - anchor
                    # Scale the vector
                    kps[idx] = anchor + vec * scale

        # Apply offset (relative to reference dimensions)
        if offset_x != 0.0 or offset_y != 0.0:
            offset = np.array([offset_x * ref_width, offset_y * ref_height])
            for idx in indices:
                if idx < len(kps):
                    kps[idx] += offset

    def process(self, pose_data, head_scale, head_offset_x, head_offset_y,
                neck_length_scale, neck_offset_x,
                torso_scale, torso_width_scale, torso_offset_x, torso_offset_y,
                shoulders_width_scale, shoulders_offset_y,
                left_arm_scale, left_arm_offset_x, left_arm_offset_y,
                right_arm_scale, right_arm_offset_x, right_arm_offset_y,
                left_leg_scale, left_leg_offset_x, left_leg_offset_y,
                right_leg_scale, right_leg_offset_x, right_leg_offset_y,
                left_hand_scale, left_hand_offset_x, left_hand_offset_y,
                right_hand_scale, right_hand_offset_x, right_hand_offset_y):

        import copy

        # Deep copy the pose data to avoid modifying the original
        modified_pose_data = copy.deepcopy(pose_data)

        # Process each frame
        for meta in modified_pose_data["pose_metas"]:
            width = meta.width
            height = meta.height

            # Body keypoint indices:
            # 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
            # 5: LShoulder, 6: LElbow, 7: LWrist, 8: RHip, 9: RKnee,
            # 10: RAnkle, 11: LHip, 12: LKnee, 13: LAnkle,
            # 14: REye, 15: LEye, 16: REar, 17: LEar, 18: LToe, 19: RToe

            # Calculate skeleton dimensions for relative offsets
            # Use skeleton height and width instead of image dimensions
            # This makes offsets proportional to the actual character size
            skeleton_height = np.abs(meta.kps_body[0][1] - max(meta.kps_body[10][1], meta.kps_body[13][1]))  # Head to feet
            skeleton_width = np.abs(meta.kps_body[2][0] - meta.kps_body[5][0])  # Right shoulder to left shoulder

            # Fallback to image dimensions if skeleton is not detected properly
            if skeleton_height < 1:
                skeleton_height = height * 0.5
            if skeleton_width < 1:
                skeleton_width = width * 0.3

            # Torso manipulation (neck to hips) - anchor at center of torso
            if torso_scale != 1.0 or torso_width_scale != 1.0 or torso_offset_x != 0.0 or torso_offset_y != 0.0:
                # Get torso center (between neck and hips)
                torso_center = (meta.kps_body[1] + meta.kps_body[8] + meta.kps_body[11]) / 3.0

                # Scale torso height (shoulders and hips relative to neck)
                if torso_scale != 1.0:
                    for idx in [2, 5, 8, 11]:  # Shoulders and hips
                        vec = meta.kps_body[idx] - meta.kps_body[1]  # From neck
                        meta.kps_body[idx] = meta.kps_body[1] + vec * torso_scale

                # Scale torso width (shoulders and hips relative to center)
                if torso_width_scale != 1.0:
                    shoulder_center = (meta.kps_body[2] + meta.kps_body[5]) / 2.0
                    hip_center = (meta.kps_body[8] + meta.kps_body[11]) / 2.0

                    # Scale shoulders
                    for idx in [2, 5]:
                        vec = meta.kps_body[idx] - shoulder_center
                        vec[0] *= torso_width_scale  # Only scale horizontally
                        meta.kps_body[idx] = shoulder_center + vec

                    # Scale hips
                    for idx in [8, 11]:
                        vec = meta.kps_body[idx] - hip_center
                        vec[0] *= torso_width_scale  # Only scale horizontally
                        meta.kps_body[idx] = hip_center + vec

                # Apply offset to entire torso (relative to skeleton size)
                if torso_offset_x != 0.0 or torso_offset_y != 0.0:
                    offset = np.array([torso_offset_x * skeleton_width, torso_offset_y * skeleton_height])
                    for idx in [1, 2, 5, 8, 11]:  # Neck, shoulders, hips
                        meta.kps_body[idx] += offset

            # Shoulder width and position adjustment
            if shoulders_width_scale != 1.0 or shoulders_offset_y != 0.0:
                shoulder_center = (meta.kps_body[2] + meta.kps_body[5]) / 2.0

                # Scale shoulder width
                if shoulders_width_scale != 1.0:
                    for idx in [2, 5]:
                        vec = meta.kps_body[idx] - shoulder_center
                        vec[0] *= shoulders_width_scale
                        meta.kps_body[idx] = shoulder_center + vec

                # Offset shoulders vertically (relative to skeleton height)
                if shoulders_offset_y != 0.0:
                    offset_y = shoulders_offset_y * skeleton_height
                    meta.kps_body[2][1] += offset_y
                    meta.kps_body[5][1] += offset_y

            # Neck manipulation - scale neck length and apply horizontal offset
            if neck_length_scale != 1.0 or neck_offset_x != 0.0:
                neck_point = meta.kps_body[1].copy()

                # Scale neck length: move head away from/towards neck point
                if neck_length_scale != 1.0:
                    head_indices = [0, 14, 15, 16, 17]  # Nose, eyes, ears
                    for idx in head_indices:
                        if idx < len(meta.kps_body):
                            # Vector from neck to head keypoint
                            vec = meta.kps_body[idx] - neck_point
                            # Scale the vector to lengthen/shorten neck
                            meta.kps_body[idx] = neck_point + vec * neck_length_scale

                # Apply horizontal offset (head lean/tilt)
                if neck_offset_x != 0.0:
                    offset_x = neck_offset_x * skeleton_width
                    head_indices = [0, 14, 15, 16, 17]
                    for idx in head_indices:
                        if idx < len(meta.kps_body):
                            meta.kps_body[idx][0] += offset_x

            # Head manipulation - scale facial features relative to nose, not neck
            # This prevents neck stretching
            if head_scale != 1.0 or head_offset_x != 0.0 or head_offset_y != 0.0:
                # Store original nose position
                original_nose = meta.kps_body[0].copy()

                # Scale facial features (eyes, ears) relative to nose (not neck)
                if head_scale != 1.0:
                    face_indices = [14, 15, 16, 17]  # Eyes and ears only
                    for idx in face_indices:
                        if idx < len(meta.kps_body):
                            vec = meta.kps_body[idx] - original_nose
                            meta.kps_body[idx] = original_nose + vec * head_scale

                # Apply offset to entire head (nose + facial features) AND neck
                # This keeps the neck-to-head connection stable (relative to skeleton size)
                if head_offset_x != 0.0 or head_offset_y != 0.0:
                    offset = np.array([head_offset_x * skeleton_width, head_offset_y * skeleton_height])
                    head_and_neck_indices = [0, 1, 14, 15, 16, 17]  # Include neck!
                    for idx in head_and_neck_indices:
                        if idx < len(meta.kps_body):
                            meta.kps_body[idx] += offset

            # Calculate limb-specific dimensions for more precise control
            left_arm_length = np.linalg.norm(meta.kps_body[7] - meta.kps_body[5])  # Wrist to shoulder
            right_arm_length = np.linalg.norm(meta.kps_body[4] - meta.kps_body[2])
            left_leg_length = np.linalg.norm(meta.kps_body[13] - meta.kps_body[11])  # Ankle to hip
            right_leg_length = np.linalg.norm(meta.kps_body[10] - meta.kps_body[8])

            # Left arm - store wrist position before manipulation
            left_wrist_before = meta.kps_body[7].copy()
            left_arm_indices = [6, 7]  # Elbow and wrist
            self.manipulate_keypoints(meta.kps_body, left_arm_indices, 5, left_arm_scale,
                                     left_arm_offset_x, left_arm_offset_y, left_arm_length, left_arm_length)
            left_wrist_delta = meta.kps_body[7] - left_wrist_before

            # Right arm - store wrist position before manipulation
            right_wrist_before = meta.kps_body[4].copy()
            right_arm_indices = [3, 4]  # Elbow and wrist
            self.manipulate_keypoints(meta.kps_body, right_arm_indices, 2, right_arm_scale,
                                     right_arm_offset_x, right_arm_offset_y, right_arm_length, right_arm_length)
            right_wrist_delta = meta.kps_body[4] - right_wrist_before

            # Left leg (hip, knee, ankle, toe) - anchor at hip
            left_leg_indices = [12, 13, 18]  # Don't include hip (11) in manipulation
            self.manipulate_keypoints(meta.kps_body, left_leg_indices, 11, left_leg_scale,
                                     left_leg_offset_x, left_leg_offset_y, left_leg_length, left_leg_length)

            # Right leg (hip, knee, ankle, toe) - anchor at hip
            right_leg_indices = [9, 10, 19]  # Don't include hip (8) in manipulation
            self.manipulate_keypoints(meta.kps_body, right_leg_indices, 8, right_leg_scale,
                                     right_leg_offset_x, right_leg_offset_y, right_leg_length, right_leg_length)

            # Hand size for relative hand offsets
            hand_size = skeleton_width * 0.15  # Hands are roughly 15% of skeleton width

            # Left hand - translate to follow wrist, then manipulate
            if meta.kps_lhand is not None and len(meta.kps_lhand) > 0:
                # First translate entire hand to follow the wrist
                meta.kps_lhand += left_wrist_delta

                # Then apply hand-specific scaling and offset (relative to hand size)
                left_hand_indices = list(range(1, len(meta.kps_lhand)))  # All except wrist
                self.manipulate_keypoints(meta.kps_lhand, left_hand_indices, 0, left_hand_scale,
                                         left_hand_offset_x, left_hand_offset_y, hand_size, hand_size)

            # Right hand - translate to follow wrist, then manipulate
            if meta.kps_rhand is not None and len(meta.kps_rhand) > 0:
                # First translate entire hand to follow the wrist
                meta.kps_rhand += right_wrist_delta

                # Then apply hand-specific scaling and offset (relative to hand size)
                right_hand_indices = list(range(1, len(meta.kps_rhand)))  # All except wrist
                self.manipulate_keypoints(meta.kps_rhand, right_hand_indices, 0, right_hand_scale,
                                         right_hand_offset_x, right_hand_offset_y, hand_size, hand_size)

        # Also update pose_metas_original if it exists
        if "pose_metas_original" in modified_pose_data:
            for meta_orig in modified_pose_data["pose_metas_original"]:
                width = meta_orig["width"]
                height = meta_orig["height"]

                # Convert to unnormalized coordinates for manipulation
                kps_body = meta_orig["keypoints_body"][:, :2] * np.array([width, height])
                kps_lhand = meta_orig["keypoints_left_hand"][:, :2] * np.array([width, height])
                kps_rhand = meta_orig["keypoints_right_hand"][:, :2] * np.array([width, height])

                # Calculate skeleton dimensions for relative offsets
                skeleton_height = np.abs(kps_body[0][1] - max(kps_body[10][1], kps_body[13][1]))
                skeleton_width = np.abs(kps_body[2][0] - kps_body[5][0])
                if skeleton_height < 1:
                    skeleton_height = height * 0.5
                if skeleton_width < 1:
                    skeleton_width = width * 0.3

                # Torso manipulation
                if torso_scale != 1.0 or torso_width_scale != 1.0 or torso_offset_x != 0.0 or torso_offset_y != 0.0:
                    # Scale torso height
                    if torso_scale != 1.0:
                        for idx in [2, 5, 8, 11]:
                            vec = kps_body[idx] - kps_body[1]
                            kps_body[idx] = kps_body[1] + vec * torso_scale

                    # Scale torso width
                    if torso_width_scale != 1.0:
                        shoulder_center = (kps_body[2] + kps_body[5]) / 2.0
                        hip_center = (kps_body[8] + kps_body[11]) / 2.0
                        for idx in [2, 5]:
                            vec = kps_body[idx] - shoulder_center
                            vec[0] *= torso_width_scale
                            kps_body[idx] = shoulder_center + vec
                        for idx in [8, 11]:
                            vec = kps_body[idx] - hip_center
                            vec[0] *= torso_width_scale
                            kps_body[idx] = hip_center + vec

                    # Apply offset (relative to skeleton size)
                    if torso_offset_x != 0.0 or torso_offset_y != 0.0:
                        offset = np.array([torso_offset_x * skeleton_width, torso_offset_y * skeleton_height])
                        for idx in [1, 2, 5, 8, 11]:
                            kps_body[idx] += offset

                # Shoulder manipulation
                if shoulders_width_scale != 1.0 or shoulders_offset_y != 0.0:
                    shoulder_center = (kps_body[2] + kps_body[5]) / 2.0
                    if shoulders_width_scale != 1.0:
                        for idx in [2, 5]:
                            vec = kps_body[idx] - shoulder_center
                            vec[0] *= shoulders_width_scale
                            kps_body[idx] = shoulder_center + vec
                    if shoulders_offset_y != 0.0:
                        offset_y = shoulders_offset_y * skeleton_height
                        kps_body[2][1] += offset_y
                        kps_body[5][1] += offset_y

                # Neck manipulation
                if neck_length_scale != 1.0 or neck_offset_x != 0.0:
                    neck_point = kps_body[1].copy()

                    # Scale neck length
                    if neck_length_scale != 1.0:
                        head_indices = [0, 14, 15, 16, 17]
                        for idx in head_indices:
                            vec = kps_body[idx] - neck_point
                            kps_body[idx] = neck_point + vec * neck_length_scale

                    # Apply horizontal offset
                    if neck_offset_x != 0.0:
                        offset_x = neck_offset_x * skeleton_width
                        head_indices = [0, 14, 15, 16, 17]
                        for idx in head_indices:
                            kps_body[idx][0] += offset_x

                # Head - prevent neck stretching
                if head_scale != 1.0 or head_offset_x != 0.0 or head_offset_y != 0.0:
                    original_nose = kps_body[0].copy()

                    # Scale facial features relative to nose
                    if head_scale != 1.0:
                        face_indices = [14, 15, 16, 17]
                        for idx in face_indices:
                            vec = kps_body[idx] - original_nose
                            kps_body[idx] = original_nose + vec * head_scale

                    # Offset head and neck together (relative to skeleton size)
                    if head_offset_x != 0.0 or head_offset_y != 0.0:
                        offset = np.array([head_offset_x * skeleton_width, head_offset_y * skeleton_height])
                        head_and_neck_indices = [0, 1, 14, 15, 16, 17]
                        for idx in head_and_neck_indices:
                            kps_body[idx] += offset

                # Calculate limb lengths for relative offsets
                left_arm_length = np.linalg.norm(kps_body[7] - kps_body[5])
                right_arm_length = np.linalg.norm(kps_body[4] - kps_body[2])
                left_leg_length = np.linalg.norm(kps_body[13] - kps_body[11])
                right_leg_length = np.linalg.norm(kps_body[10] - kps_body[8])

                # Left arm - track wrist movement
                left_wrist_before = kps_body[7].copy()
                left_arm_indices = [6, 7]
                self.manipulate_keypoints(kps_body, left_arm_indices, 5, left_arm_scale,
                                         left_arm_offset_x, left_arm_offset_y, left_arm_length, left_arm_length)
                left_wrist_delta = kps_body[7] - left_wrist_before

                # Right arm - track wrist movement
                right_wrist_before = kps_body[4].copy()
                right_arm_indices = [3, 4]
                self.manipulate_keypoints(kps_body, right_arm_indices, 2, right_arm_scale,
                                         right_arm_offset_x, right_arm_offset_y, right_arm_length, right_arm_length)
                right_wrist_delta = kps_body[4] - right_wrist_before

                # Left leg
                left_leg_indices = [12, 13, 18]
                self.manipulate_keypoints(kps_body, left_leg_indices, 11, left_leg_scale,
                                         left_leg_offset_x, left_leg_offset_y, left_leg_length, left_leg_length)

                # Right leg
                right_leg_indices = [9, 10, 19]
                self.manipulate_keypoints(kps_body, right_leg_indices, 8, right_leg_scale,
                                         right_leg_offset_x, right_leg_offset_y, right_leg_length, right_leg_length)

                # Hand size for relative offsets
                hand_size = skeleton_width * 0.15

                # Left hand - translate to follow wrist
                if kps_lhand is not None and len(kps_lhand) > 0:
                    kps_lhand += left_wrist_delta
                    left_hand_indices = list(range(1, len(kps_lhand)))
                    self.manipulate_keypoints(kps_lhand, left_hand_indices, 0, left_hand_scale,
                                             left_hand_offset_x, left_hand_offset_y, hand_size, hand_size)

                # Right hand - translate to follow wrist
                if kps_rhand is not None and len(kps_rhand) > 0:
                    kps_rhand += right_wrist_delta
                    right_hand_indices = list(range(1, len(kps_rhand)))
                    self.manipulate_keypoints(kps_rhand, right_hand_indices, 0, right_hand_scale,
                                             right_hand_offset_x, right_hand_offset_y, hand_size, hand_size)

                # Normalize back
                meta_orig["keypoints_body"][:, :2] = kps_body / np.array([width, height])
                meta_orig["keypoints_left_hand"][:, :2] = kps_lhand / np.array([width, height])
                meta_orig["keypoints_right_hand"][:, :2] = kps_rhand / np.array([width, height])

        return (modified_pose_data,)

NODE_CLASS_MAPPINGS = {
    "OnnxDetectionModelLoader": OnnxDetectionModelLoader,
    "PoseAndFaceDetection": PoseAndFaceDetection,
    "DrawViTPose": DrawViTPose,
    "PoseRetargetPromptHelper": PoseRetargetPromptHelper,
    "PoseBoneManipulation": PoseBoneManipulation,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnnxDetectionModelLoader": "ONNX Detection Model Loader",
    "PoseAndFaceDetection": "Pose and Face Detection",
    "DrawViTPose": "Draw ViT Pose",
    "PoseRetargetPromptHelper": "Pose Retarget Prompt Helper",
    "PoseBoneManipulation": "Pose Bone Manipulation",
}
