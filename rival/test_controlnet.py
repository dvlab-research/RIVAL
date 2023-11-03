import sys
sys.path += ['.']
import json
import argparse
import types
import torch
import os

from icecream import ic
from PIL import Image
from ddim_inversion import Inversion, load_512
from attention_forward import new_forward
from diffusers.models.attention import CrossAttention
from diffusers import ControlNetModel, DDIMScheduler
from sd_pipeline_controlnet import RIVALStableDiffusionControlNetPipeline

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

parser = argparse.ArgumentParser()
# load the config.
parser.add_argument("--inf_config", type=str, default="configs/rival_controlnet.json")
parser.add_argument("--img_config", type=str, default="assets/images/configs_controlnet.json")
parser.add_argument("--inner_round", type=int, default=1, help="number of images per reference")
parser.add_argument("--exp_folder", type=str, default="out/controlnet_exps")
parser.add_argument("--pretrained_model_path", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--is_half", type=bool, default=False)
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

if args.is_half:
    dtype=torch.float16
else:
    dtype=torch.float32

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=dtype
)

ldm_controlnet = RIVALStableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype
)
ldm_controlnet.scheduler = scheduler

count = 0
for module in ldm_controlnet.unet.modules():
    if isinstance(module, CrossAttention):
        count += 1
        # use a placeholder function for the original forward.
        module.ori_forward = module.forward
        module.cfg = attn_cfgs.copy()
        module.init_step = 1000
        module.step_size = module.init_step // cfgs.copy()["inference"]["ddim_step"]
        module.t_align = module.cfg["t_align"]
        module.forward = types.MethodType(new_forward, module)

ldm_controlnet.enable_model_cpu_offload()
ldm_controlnet.enable_xformers_memory_efficient_attention()

inversion = Inversion(ldm_controlnet, GUIDANCE_SCALE, NUM_DDIM_STEPS, INVERT_STEPS)
os.makedirs(os.path.join(args.exp_folder), exist_ok=True)

with open(args.img_config) as f:
    cfg = json.load(f)
    image_exps = cfg["image_exps"]
    image_root = cfg["image_root"]
    prompt_postfix = cfg["prompt_postfix"]
    neg_prompt = cfg["neg_prompt"]

current_state = "canny"
for cfg_item in image_exps:
    mode = cfg_item['mode']
    control_image_path = os.path.join(image_root, cfg_item['control_image_path'])
    control_image = Image.open(control_image_path).convert("RGB")
    control_image = control_image.resize((512, 512), Image.LANCZOS)
    
    if mode != current_state:
        ldm_controlnet.controlnet = ControlNetModel.from_pretrained(
            f"lllyasviel/sd-controlnet-{mode}", torch_dtype=dtype
        ).to(device)
        ic(f"load controlnet model: lllyasviel/sd-controlnet-{mode}")
        current_state = mode
        
    exp_name = cfg_item['exp_name']
    prompt = cfg_item['prompt'] + prompt_postfix
    image_path = os.path.join(image_root, cfg_item['image_path'])
    prompts = [prompt]
    ic(prompt)

    if IS_NULL_PROMPT:
        prompt = ""
    
    inversion.init_prompt(prompt)  
    image_gt = load_512(image_path)
    rec_, x_ts = inversion.ddim_inversion(image_gt, dtype=dtype)
    x_t = x_ts[-1]
    png_name = os.path.join(args.exp_folder, exp_name + '.png')
    
    generator = torch.Generator(device=device)

    if IS_NULL_PROMPT:
        prompts = ["", prompts[0]]
    else:
        prompts = [prompts[0], prompts[0]]
    for m in range(args.inner_round):
        idx = torch.randperm(x_t.nelement()//4)
        x_t_append_norm = x_t.view(1, 4, -1)[:, :, idx].view(x_t.size())
        x_t_in = torch.cat([x_t, x_t_append_norm], dim=0)
        
        with torch.no_grad():
            images = ldm_controlnet(
                prompts,
                control_image,
                negative_prompt=["", neg_prompt],
                generator=generator,
                latents=x_t_in,
                num_images_per_prompt=1,
                num_inference_steps = NUM_DDIM_STEPS,
                guidance_scale = GUIDANCE_SCALE,
                is_adain = True,
                chain = x_ts,
                t_early = T_EARLY,
                # controlnet_conditioning_scale=0.6, # uncomment to contorl guidance.
            ).images
        
        grid = image_grid(images, 1, 2)
        # save this image
        grid.save(png_name[:-4] + f"_{m}.png")

