import json, os, gc
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
BASE = "/root/autodl-tmp/checkpoints/openvla-7b"

img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
prompt = "In: What action should the robot take to pick up the black bowl between the plate and the ramekin and place it on the plate?\nOut:"

for name, ckpt, key in [("OURS", OURS, "libero_spatial_no_noops"), ("BASE", BASE, "bridge_orig")]:
    proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to("cuda:0")
    model.eval()
    sp = os.path.join(ckpt, "dataset_statistics.json")
    if os.path.isfile(sp):
        model.norm_stats = json.load(open(sp))
    inputs = proc(prompt, img).to("cuda:0", dtype=torch.bfloat16)
    with torch.inference_mode():
        act = model.predict_action(**inputs, unnorm_key=key, do_sample=False)
    print(name, "->", np.round(act, 4).tolist(), flush=True)
    del model, proc
    torch.cuda.empty_cache()
    gc.collect()

print("DONE")
