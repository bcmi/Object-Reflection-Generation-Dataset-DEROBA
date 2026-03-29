import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
import torch.nn.functional as F
from cldm.model import create_model, load_state_dict
import random
random.seed(42)

def trans_tensor2img(grid_, if_mask=False):
    if if_mask:
        grid = (grid_ > 0.5).float()
    else:
        grid = (grid_ + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    result_img = Image.fromarray(grid)
    return result_img

def process_single_image(composite_path, mask_path):
 
    width, height = 512, 512
    
    reflectionfree_img = cv2.imread(composite_path)
    reflectionfree_img = cv2.resize(reflectionfree_img, (width, height))
    
    object_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    object_mask = cv2.resize(object_mask, (width, height))

    reflection_img = cv2.imread(composite_path)
    reflection_img = cv2.resize(reflection_img, (width, height))

    _, fg_instance_thresh = cv2.threshold(object_mask, 128, 255, cv2.THRESH_BINARY)
    contours_instance, _ = cv2.findContours(fg_instance_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    merged_contour_points_instance = np.concatenate(contours_instance)
    rect_instance = cv2.minAreaRect(merged_contour_points_instance)
    (x, y), (w, h), theta = rect_instance
    if w < h:
        w, h = h, w
        theta = theta + 90
    bbx_instance = np.array([x, y, w+1, h+1, theta]).astype(int)
    bbx_instance = torch.tensor(bbx_instance)

    reflectionfree_img = cv2.cvtColor(reflectionfree_img, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(reflection_img, cv2.COLOR_BGR2RGB) 

    source = np.concatenate((reflectionfree_img, object_mask[:, :, np.newaxis]), axis=-1)
    cls_input = np.concatenate((reflectionfree_img, object_mask[:, :, np.newaxis]), axis=-1)
    
    cls_input = cls_input.astype(np.float32) / 255.0
    source = source.astype(np.float32) / 255.0
    object_mask = object_mask.astype(np.float32) / 255.0
    target = (target.astype(np.float32) / 127.5) - 1.0

    reflection_img_ = cv2.imread(composite_path)
    reflection_img_ = cv2.resize(reflection_img_, (256, 256))
    target_ = cv2.cvtColor(reflection_img_, cv2.COLOR_BGR2RGB)
    target_ = (target_.astype(np.float32) / 127.5) - 1.0
    
    reflectionfree_img_ = cv2.imread(composite_path)
    reflectionfree_img_ = cv2.resize(reflectionfree_img_, (256, 256))
    reflectionfree_img_ = cv2.cvtColor(reflectionfree_img_, cv2.COLOR_BGR2RGB)
    reflectionfree_img_ = (reflectionfree_img_.astype(np.float32) / 127.5) - 1.0

    object_mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    object_mask_ = cv2.resize(object_mask_, (256, 256))
    object_mask_ = object_mask_.astype(np.float32) / 255.0
    
    mask_embeddings = torch.zeros((64, 2048), dtype=torch.float32)
    bbx_region = torch.zeros((512, 512), dtype=torch.float32)


    data_dict = dict(
        jpg=torch.from_numpy(target), 
        cls=torch.from_numpy(cls_input), 
        fg=bbx_instance, 
        bbx=bbx_region, 
        embeddings=mask_embeddings, 
        txt='', 
        hint=torch.from_numpy(source), 
        objectmask=torch.from_numpy(object_mask),
        reflectionfree_img_=torch.from_numpy(reflectionfree_img_), 
        object_mask_=torch.from_numpy(object_mask_)
    )
    return data_dict

def main():
    parser = argparse.ArgumentParser('Single Image Inference')
    parser.add_argument('--composite_image', type=str, default='test_examples/composite.png', help='Path to composite image')
    parser.add_argument('--foreground_mask', type=str, default='test_examples/mask.png', help='Path to foreground mask')
    parser.add_argument('--checkpoint_path', type=str, default='models/Reflection_cldm.ckpt')
    parser.add_argument('--model_dir', type=str, default='models/cldm_v15.yaml')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_generate', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='output/')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading model...")
    model = create_model(args.model_dir).cpu()
    model.load_state_dict(load_state_dict(args.checkpoint_path, location='cuda'), strict=False)
    model = model.to(device)
    model.eval()

    print("Processing input data...")
    raw_data = process_single_image(args.composite_image, args.foreground_mask)
    
    batch = {}
    n = args.num_generate
    for key, value in raw_data.items():
        if isinstance(value, torch.Tensor):
            if len(value.shape) == 1:
                batch[key] = value.unsqueeze(0).repeat(n, 1).to(device)
            elif len(value.shape) == 2:
                batch[key] = value.unsqueeze(0).repeat(n, 1, 1).to(device)
            elif len(value.shape) == 3:
                batch[key] = value.unsqueeze(0).repeat(n, 1, 1, 1).to(device)
            else:
                batch[key] = value.to(device)
        elif isinstance(value, str):
            batch[key] = [value] * n

    print(f"Generating {n} results...")
    with torch.no_grad():
        images = model.log_images(batch, N=n, use_x_T=True)

    pic_name = os.path.splitext(os.path.basename(args.composite_image))[0]
    
    for k in images:
        if isinstance(images[k], torch.Tensor):
            images[k] = torch.clamp(images[k].detach().cpu(), -1., 1.)

    for i in range(n):

        sample_key = 'samples_cfg_scale_9.00'
        if sample_key in images:
            result_tensor = images[sample_key][i].unsqueeze(0)
            result_rescaled = F.interpolate(result_tensor, size=(512, 512), mode='bilinear', align_corners=True)
            
            result_pil = trans_tensor2img(result_rescaled.squeeze(0))
            save_path = os.path.join(args.save_dir, f"{pic_name}_gen_{i}.png")
            result_pil.save(save_path)
            print(f"Saved: {save_path}")

    print("Done.")

if __name__ == '__main__':
    main()