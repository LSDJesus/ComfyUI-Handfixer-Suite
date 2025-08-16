import torch
import numpy as np
from PIL import Image
from .utils import MediapipeEngine
import comfy.model_management
import comfy.sample
import comfy.utils

# A global variable to hold our engine instance so it's not re-initialized every time
ENGINE_INSTANCE = None

class MediaPipeDetailer:
    """The all-in-one MediaPipe Detailer Suite node."""

    def __init__(self):
        # This gets called when the node is created.
        pass

    @classmethod
    def INPUT_TYPES(s):
        """Defines the inputs, outputs, and widgets for the node."""
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": (["hands", "face_mesh", "pose", "holistic", "selfie", "feet", "eyes", "mouth"],),
                "sort_by": (["Confidence", "Area: Largest to Smallest", "Position: Left to Right", "Position: Top to Bottom", "Position: Nearest to Center"],),
                "max_objects": ("INT", {"default": 10, "min": 1, "max": 10, "step": 1}),
                "confidence": ("FLOAT", {"default": 0.30, "min": 0.1, "max": 1.0, "step": 0.05}),
                "mask_padding": ("INT", {"default": 35, "min": 0, "max": 200, "step": 1}),
                "mask_blur": ("INT", {"default": 6, "min": 0, "max": 100, "step": 1}),
                "detail_positive_prompt": ("STRING", {"multiline": True, "default": "A high-resolution photograph with realistic details, high quality"}),
                "detail_negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality, deformed"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "flux_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            },
            "optional": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "MASK", "IMAGE", "CLIP", "VAE")
    RETURN_NAMES = ("POSITIVE", "NEGATIVE", "LATENT", "MASK", "IMAGE_PASSTHROUGH", "CLIP_PASSTHROUGH", "VAE_PASSTHROUGH")
    FUNCTION = "process"
    CATEGORY = "HandFixerSuite"

    def process(self, image, model_type, sort_by, max_objects, confidence, mask_padding, mask_blur, detail_positive_prompt, detail_negative_prompt, denoise, flux_guidance_strength, clip=None, vae=None):
        global ENGINE_INSTANCE
        if ENGINE_INSTANCE is None:
            ENGINE_INSTANCE = MediapipeEngine()

        # --- Prepare data ---
        positive_cond, negative_cond, masked_latent = None, None, None
        
        # image tensor is (Batch, H, W, C)
        np_image = (image[0].numpy() * 255).astype(np.uint8)

        # --- Run Mask Generation ---
        options = {
            'model_type': model_type,
            'sort_by': sort_by,
            'max_objects': max_objects,
            'confidence': confidence,
            'mask_padding': mask_padding,
            'mask_blur': mask_blur,
        }
        final_mask_np = ENGINE_INSTANCE.process_and_create_mask(np_image, options)
        # Convert mask back to a tensor (H, W) -> (1, H, W)
        final_mask_tensor = torch.from_numpy(final_mask_np.astype(np.float32) / 255.0).unsqueeze(0)

        # --- Conditional Logic for Conditioning and Latent ---
        has_prompt_text = detail_positive_prompt != "" or detail_negative_prompt != ""

        if clip is not None:
            # Replicate CLIPTextEncode logic
            pos_tokens = clip.tokenize(detail_positive_prompt)
            pos_cond, _ = clip.encode_from_tokens(pos_tokens, return_pooled=False)
            
            neg_tokens = clip.tokenize(detail_negative_prompt)
            neg_cond, _ = clip.encode_from_tokens(neg_tokens, return_pooled=False)
            
            # Replicate FluxGuidance logic
            positive_cond = [[pos_cond, {"guidance_strength": flux_guidance_strength}]]
            negative_cond = [[neg_cond, {}]]

        elif has_prompt_text:
            print(f"\n[MediaPipeDetailer] Warning: Text was found in a prompt field, but no CLIP model was connected. The prompt will be ignored.")

        if vae is not None:
            # Replicate VAEEncodeForInpaint logic
            mask_for_latent = final_mask_tensor.to(vae.device)
            if mask_for_latent.shape[2] != image.shape[2] or mask_for_latent.shape[1] != image.shape[1]:
                mask_for_latent = comfy.utils.common_upscale(mask_for_latent.movedim(0, -1), image.shape[2], image.shape[1], "bicubic", "center").movedim(-1, 0)
            
            # ComfyUI's VAE encode expects image in (B, C, H, W), but our node gets (B, H, W, C)
            image_for_latent = image.permute(0, 3, 1, 2)
            latent = vae.encode(image_for_latent[:,:3,:,:])
            
            # The mask for the latent must be a single channel and match the latent's HxW
            latent_mask = final_mask_tensor.mean(dim=0, keepdim=True).repeat(latent.shape[0], 1, 1, 1)
            latent_mask = comfy.utils.common_upscale(latent_mask, latent.shape[3], latent.shape[2], "bicubic", "center")
            masked_latent = {"samples": latent, "noise_mask": latent_mask}
            
        # --- Return all outputs ---
        # The tuple must match RETURN_TYPES in order
        return (positive_cond, negative_cond, masked_latent, final_mask_tensor, image, clip, vae)