# experiment for re-distribution of x_t as a random noise.
import sys
sys.path += ['.']
import json
import argparse
import os
import torch
import utils
import types
from PIL import Image
from diffusers import DDIMScheduler
from ddim_inversion import Inversion, load_512
from diffusers.models.attention import CrossAttention
from sd_pipeline_img import RIVALStableDiffusionPipeline
from attention_forward import new_forward
from icecream import ic
from torchvision import transforms

convert_tensor = transforms.ToTensor()
    
parser = argparse.ArgumentParser()
# load the config.
parser.add_argument("--inf_config", type=str, default="configs/rival_variation.json")
parser.add_argument("--img_config", type=str, default="assets/images/configs_variation.json")
parser.add_argument("--inner_round", type=int, default=1, help="number of images per reference")
parser.add_argument("--exp_folder", type=str, default="out/variation_exps")
parser.add_argument("--pretrained_model_path", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--is_half", type=bool, default=False)
parser.add_argument("--is_editing", type=bool, default=False)
parser.add_argument("--editing_early_steps", type=int, default=1000)
args = parser.parse_args()
ic(args)

with open(args.inf_config) as openfile:
    cfgs = json.load(openfile)
    ic(cfgs)
    attn_cfgs = cfgs["self_attn"]

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)

################################################# Config ###################################################

NUM_DDIM_STEPS = cfgs["inference"]['ddim_step']
INVERT_STEPS = cfgs["inference"]['invert_step']
GUIDANCE_SCALE = cfgs["inference"]['cfg']
IS_NULL_PROMPT = cfgs["inference"]['is_null_prompt']
T_EARLY = cfgs["inference"]['t_early']
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

################################################# Config ###################################################
ic(args.pretrained_model_path)
if args.is_half:
    dtype=torch.float16
else:
    dtype=torch.float32
    
ldm_stable = RIVALStableDiffusionPipeline.from_pretrained(
    args.pretrained_model_path, torch_dtype=dtype
).to(device)
ldm_stable.scheduler = scheduler

for module in ldm_stable.unet.modules():
    if isinstance(module, CrossAttention):
        # use a placeholder function for the original forward.
        module.ori_forward = module.forward
        module.cfg = attn_cfgs.copy()
        module.init_step = 1000
        module.step_size = module.init_step // cfgs.copy()["inference"]["ddim_step"]
        module.t_align = module.cfg["t_align"]
        module.editing_early_steps = args.editing_early_steps
        module.forward = types.MethodType(new_forward, module)
        
ldm_stable.enable_model_cpu_offload()
ldm_stable.enable_xformers_memory_efficient_attention()

inversion = Inversion(ldm_stable, GUIDANCE_SCALE, NUM_DDIM_STEPS, INVERT_STEPS)
os.makedirs(os.path.join(args.exp_folder), exist_ok=True)

with open(args.img_config) as f:
    cfg = json.load(f)
    image_exps = cfg["image_exps"]
    image_root = cfg["image_root"]
    prompt_postfix = cfg["prompt_postfix"]
    neg_prompt = cfg["neg_prompt"]
    
for cfg_item in image_exps:
    exp_name = cfg_item['exp_name']
    prompt = cfg_item['prompt'] + prompt_postfix
    
    if args.is_editing:
        prompt_ori = cfg_item['prompt_ori']
        prompt_edit = cfg_item['prompt']
        
    image_path = os.path.join(image_root, cfg_item['image_path'])
    prompts = [prompt]
    ic(prompt)
        
    if IS_NULL_PROMPT:
        prompt = ""
    elif args.is_editing:
        prompt = prompt_ori
        
    inversion.init_prompt(prompt)  
    image_gt = load_512(image_path)
    rec_, x_ts = inversion.ddim_inversion(image_gt, dtype=dtype)
    x_t = x_ts[-1]
    png_name = os.path.join(args.exp_folder, exp_name + '.png')
    
    generator = torch.Generator(device=device)

    if IS_NULL_PROMPT:
        prompts = ["", prompts[0]]
        if args.is_editing:
            prompts = ["", prompt_edit]
    else:
        prompts = [prompts[0], prompts[0]]
        if args.is_editing:
            prompts = [prompt_ori, prompt_edit]
            
    # check whether use inpainting:
    mask = None
    if 'mask_path' in cfg_item:
        mask_path = os.path.join(image_root, cfg_item['mask_path'])
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((64, 64), resample=Image.LANCZOS)
        # convert to PIL image to binary mode
        mask = mask.point(lambda x: 0 if x < 128 else 255, '1')
        mask = convert_tensor(mask)[0]
        
    for m in range(args.inner_round):
        if args.is_editing:
            x_t_in = torch.cat([x_t, x_t], dim=0)
        else:
            idx = torch.randperm(x_t.nelement()//4)
            x_t_append_norm = x_t.view(1, 4, -1)[:, :, idx].view(x_t.size())
            x_t_in = torch.cat([x_t, x_t_append_norm], dim=0)
        
        # for inpainting.
        if mask is not None:
            masked_latent = x_t.clone()[:, :, mask == 0].view(1, 4, -1) 
            idx = torch.randperm(masked_latent.nelement()//4)
            masked_latent = masked_latent[:, :, idx].view(masked_latent.size())
            x_t_in[1, :, mask == 0] = masked_latent
        
        with torch.no_grad():
            images = ldm_stable(
                prompts,
                negative_prompt=["", neg_prompt],
                generator=generator,
                latents=x_t_in,
                num_images_per_prompt=1,
                num_inference_steps = NUM_DDIM_STEPS,
                guidance_scale = GUIDANCE_SCALE,
                is_adain = True,
                chain = x_ts,
                t_early = T_EARLY,
                output_type = 'np',
                inpaint_mask = mask,
            ).images
        
        utils.save_images([images[1]*255], name=png_name[:-4] + f"_{m}.png", offset_ratio=0.0)