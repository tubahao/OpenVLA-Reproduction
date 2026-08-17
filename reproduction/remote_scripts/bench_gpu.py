import time
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

CKPT = "/root/autodl-tmp/checkpoints/openvla-7b"

torch.cuda.init()
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    CKPT,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")
model.eval()

img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
prompt = "In: What action should the robot take to pick up the red block?\nOut:"
inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)

with torch.inference_mode():
    for _ in range(3):
        model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
torch.cuda.synchronize()

N = 30
t0 = time.time()
with torch.inference_mode():
    for _ in range(N):
        model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
torch.cuda.synchronize()
dt = time.time() - t0

stats = {
    "model": "openvla-7b",
    "params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
    "dtype": "bfloat16",
    "num_runs": N,
    "total_s": round(dt, 2),
    "hz": round(N / dt, 2),
    "ms_per_action": round(dt / N * 1000, 1),
    "vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    "vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
    "gpu_name": torch.cuda.get_device_name(0),
}
print(json.dumps(stats, indent=2), flush=True)
