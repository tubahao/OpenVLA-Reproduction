import json, os
import torch
import numpy as np
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

OURS = "/root/autodl-tmp/exp/libero_spatial_lora/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OFFICIAL = "/root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial"

img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
prompts = [
    "In: What action should the robot take to pick up the black bowl between the plate and the ramekin and place it on the plate?\nOut:",
    "In: What action should the robot take to pick up the black bowl on the wooden cabinet and place it on the plate?\nOut:",
]

def pick_unnorm(model, keys):
    for k in keys:
        if k in model.norm_stats:
            return k
    return None

for name, ckpt in [("OURS", OURS), ("OFFICIAL", OFFICIAL)]:
    proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to("cuda:0")
    model.eval()
    with open(os.path.join(ckpt, "dataset_statistics.json")) as f:
        model.norm_stats = json.load(f)
    key = pick_unnorm(model, ["libero_spatial", "libero_spatial_no_noops"])
    print("MODEL", name, "unnorm_key", key)
    for pr in prompts:
        inputs = proc(pr, img).to("cuda:0", dtype=torch.bfloat16)
        with torch.inference_mode():
            act = model.predict_action(**inputs, unnorm_key=key, do_sample=False)
        print(name, pr.split(" to ")[1][:50], "->", np.round(act, 3).tolist())

print("DONE")
