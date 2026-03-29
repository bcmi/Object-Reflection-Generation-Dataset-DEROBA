import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import os


class TrainDataset(Dataset):
    def __init__(self, data_file_path, device=None):
        self.device = device
        
        self.deroba_root = data_file_path
        with open(os.path.join(self.deroba_root, 'train.txt'), 'r') as file:
            deroba_data = [line.rstrip('\n') for line in file]
        self.data = [os.path.join(self.deroba_root, 'composite_image', pic_name) for pic_name in deroba_data]
        
    
    def __len__(self):
        return len(self.data)
    
    def load_image(self, folder, pic_name):
        path = os.path.join(os.path.dirname(os.path.dirname(pic_name)), folder, os.path.basename(pic_name))
        return cv2.imread(path)
    
    def load_mask(self, folder, pic_name):
        path = os.path.join(os.path.dirname(os.path.dirname(pic_name)), folder, os.path.basename(pic_name))
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    def __getitem__(self, idx):
        pic_path = self.data[idx]
        
        reflectionfree_img = self.load_image('composite_image', pic_path)
        object_mask = self.load_mask('foreground_mask', pic_path)
        reflection_img = self.load_image('ground-truth_image', pic_path)
        reflection_mask_path = pic_path.replace('composite_image', 'reflection_mask')

        prompt = ''
        width, height = 512, 512
        width_mask, height_mask = 64, 64

        # Resize images
        reflectionfree_img = cv2.resize(reflectionfree_img, (width, height))
        object_mask = cv2.resize(object_mask, (width, height))
        reflection_img = cv2.resize(reflection_img, (width, height))
        
        # Process reflection mask
        reflection_mask = cv2.imread(reflection_mask_path, cv2.IMREAD_GRAYSCALE)
        reflection_mask = cv2.resize(reflection_mask, (width, height))
        
        # Process object contours
        _, fg_instance_thresh = cv2.threshold(object_mask, 128, 255, cv2.THRESH_BINARY)
        contours_instance, _ = cv2.findContours(fg_instance_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        merged_contour_points_instance = np.concatenate(contours_instance)
        rect_instance = cv2.minAreaRect(merged_contour_points_instance)
        (x, y), (w, h), theta = rect_instance
        if w < h:
            w, h = h, w
            theta = theta + 90
        bbx_instance = np.array([x, y, w+1, h+1, theta]).astype(int)

        # Process dilated reflection mask
        dilated_reflection_mask = cv2.resize(reflection_mask, (width_mask, height_mask))
        kernel = np.ones((6,6), np.uint8)
        dilated_reflection_mask = cv2.dilate(dilated_reflection_mask, kernel, iterations=1)
        
        # Convert color spaces
        reflectionfree_img = cv2.cvtColor(reflectionfree_img, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(reflection_img, cv2.COLOR_BGR2RGB)

        # Prepare inputs
        cls_input = np.concatenate((reflectionfree_img, object_mask[:, :, np.newaxis]), axis=-1)
        source = np.concatenate((reflectionfree_img, object_mask[:, :, np.newaxis]), axis=-1)

        # Normalize source images to [0, 1]
        cls_input = cls_input.astype(np.float32) / 255.0
        source = source.astype(np.float32) / 255.0
        reflection_mask = reflection_mask.astype(np.float32) / 255.0
        dilated_reflection_mask = dilated_reflection_mask.astype(np.float32) / 255.0
        object_mask = object_mask.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1]
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        # Prepare outputs
        mask_embeddings = torch.zeros((64, 2048), dtype=torch.float32)
        bbx_region = torch.zeros((512, 512), dtype=torch.float32)
        
        return dict(
            jpg=target,
            fg=bbx_instance,
            bbx=bbx_region,
            embeddings=mask_embeddings,
            txt=prompt,
            cls=cls_input,
            hint=source,
            reflectionmask=reflection_mask,
            objectmask=object_mask,
            dilated_reflection_mask=dilated_reflection_mask
        )