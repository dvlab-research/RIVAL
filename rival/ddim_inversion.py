import sys
sys.path += ['.']
import os
from typing import Union, List, Dict
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from icecream import ic

scaler = torch.cuda.amp.GradScaler()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def adain_latent(feat, cond_feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size() # [1, 4, 1, 64, 64]
    C = size[1]
    feat_var = feat.view(C, -1).var(dim=1) + eps
    feat_std = feat_var.sqrt().view(1, C, 1, 1, 1)
    feat_mean = feat.view(C, -1).mean(dim=1).view(1, C, 1, 1, 1)
    
    cond_feat_var = cond_feat.view(C, -1).var(dim=1) + eps
    cond_feat_std = cond_feat_var.sqrt().view(1, C, 1, 1, 1)
    cond_feat_mean = cond_feat.view(C, -1).mean(dim=1).view(1, C, 1, 1, 1)
    feat = (feat - feat_mean.expand(size)) / feat_std.expand(size)
    return feat * cond_feat_std.expand(size) + cond_feat_mean.expand(size)

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class Inversion:

    def __init__(self, model, GUIDANCE_SCALE, NUM_DDIM_STEPS, INVERT_STEPS):
        self.model = model
        self.GUIDANCE_SCALE = GUIDANCE_SCALE
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.INVERT_STEPS = INVERT_STEPS
        if hasattr(model, 'tokenizer'):
            self.tokenizer = self.model.tokenizer
        else:
            self.tokenizer = None
        self.model.scheduler.set_timesteps(self.NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]): # doing inversion (math)
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context): # latents: torch.Size([1, 4, 64, 64]); t: tensor(1); context: torch.Size([1, 77, 768])
        # formats are correct for video unet input; Tune-A-Video also predicts the residual
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"] # easy to out of mem
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image, dtype):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents


    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0] # len=2, uncond_embeddings
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt
        
    @torch.no_grad()
    def init_img_prompt(self, img_prompt):
        self.context = self.model._encode_prompt(img_prompt, self.model.device, 1, True)
        # self.model.disable_img_encoder()
        torch.cuda.empty_cache()
        ic(self.context.shape)
        self.prompt = "<refimg>"
    
    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(self.NUM_DDIM_STEPS), desc="inversion"):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings) # use a unet
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image, dtype=torch.float32):
        latent = self.image2latent(image, dtype=dtype)
        image_rec = self.latent2image(latent) # image: (512, 512, 3); latent: torch.Size([1, 4, 64, 64])
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

        
    def invert(self, image_path: str, prompt, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        if hasattr(self.model, 'tokenizer'):
            self.init_prompt(prompt)
        else:
            self.init_img_prompt(prompt)
        # image_gt = load_512(image_path, *offsets)
        image_gt = np.expand_dims(load_512(image_path, *offsets), 0)
        print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt) # ddim_latents is a list, like the link in Figure 3
        # image_rec refers to vq-autoencoder reconstruction
        print("Null-text optimization...")
        ic(image_rec.shape, ddim_latents[0].shape)
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon) # ddim_latents serve as GT; easy to out of mem
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings

    
