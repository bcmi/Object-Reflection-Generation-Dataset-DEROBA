import torch
import torch.nn.functional as F
from torchvision import transforms
from local_clip import FrozenCLIPImageEmbedder

model = FrozenCLIPImageEmbedder()

def process_image_with_mask(mask, image):

    mask = mask.unsqueeze(-1)  # (b, 512, 512) -> (b, 512, 512, 1)

    masked_image = image * mask  # (b, 512, 512, 3)

    flipped_vertical = torch.flip(masked_image, dims=[1])  

    flipped_horizontal = torch.flip(masked_image, dims=[2]) 

    flipped_images = torch.cat([flipped_vertical, flipped_horizontal], dim=0)  # (2b, 512, 512, 3)

    resized_images = F.interpolate(flipped_images.permute(0, 3, 1, 2), size=(224, 224), mode='bilinear', align_corners=False)
    resized_images = resized_images.permute(0, 2, 3, 1)  # (2b, 224, 224, 3)

    black_background = torch.zeros_like(resized_images)  # (2b, 224, 224, 3)

    final_images = torch.where(resized_images != 0, resized_images, black_background)

    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    final_images = final_images.permute(0, 3, 1, 2)  # (2b, 3, 224, 224)
    final_images = normalize(final_images)

    return final_images  # (2b, 3, 224, 224)

def extract_clip_features(image, model, device):
    model = model.to(device)
    with torch.no_grad():
        local_f, global_f = model(image) 
    return torch.cat([local_f, global_f], dim=1)

def generate_clip_features(img, mask):
    device = img.device
    crop_image = process_image_with_mask(mask, img[:,:,:,:3])
    clip_features = extract_clip_features(crop_image, model, device)
    b = clip_features.size(0) // 2
    split_tensors = torch.split(clip_features, b, dim=0)
    concatenated_tensor = torch.cat(split_tensors, dim=1)
    return concatenated_tensor
