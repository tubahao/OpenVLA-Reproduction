import json, os, gc
import torch
import numpy as np
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

OURS = "/root/autodl-tmp/exp/libero_spatial_lora/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OFFICIAL = "/root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial"

# --- get a real LIBERO image (task 0, first init state) ---
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
img_np = obs["agentview_image"][::-1, ::-1]  # rotate 180 like eval
img = Image.fromarray(img_np)
print("REAL_IMG", img_np.shape, img_np.mean().round(1), flush=True)

prompt = "In: What action should the robot take to pick up the black bowl between the plate and the ramekin and place it on the plate?\nOut:"

for name, ckpt, key in [("OURS", OURS, "libero_spatial_no_noops"), ("OFFICIAL", OFFICIAL, None)]:
    proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to("cuda:0")
    model.eval()
    sp = os.path.join(ckpt, "dataset_statistics.json")
    if os.path.isfile(sp):
        model.norm_stats = json.load(open(sp))
    if key is None:
        key = "libero_spatial" if "libero_spatial" in model.norm_stats else "libero_spatial_no_noops"
    inputs = proc(prompt, img).to("cuda:0", dtype=torch.bfloat16)
    with torch.inference_mode():
        act = model.predict_action(**inputs, unnorm_key=key, do_sample=False)
    print(name, "REAL_IMG_ACTION ->", np.round(act, 4).tolist(), flush=True)
    del model, proc
    torch.cuda.empty_cache()
    gc.collect()

print("DONE")
