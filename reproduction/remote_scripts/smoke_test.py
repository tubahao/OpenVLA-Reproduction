import time
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

CKPT = "/root/autodl-tmp/checkpoints/openvla-7b"

t0 = time.time()
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
print(f"[smoke] processor loaded in {time.time()-t0:.1f}s", flush=True)

model = AutoModelForVision2Seq.from_pretrained(
    CKPT,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"[smoke] model loaded in {time.time()-t0:.1f}s, {n_params/1e9:.2f}B params", flush=True)

img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
prompt = "In: What action should the robot take to pick up the red block?\nOut:"
inputs = processor(prompt, img).to(dtype=torch.bfloat16)

with torch.inference_mode():
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(f"[smoke] action shape={action.shape} value={np.round(action, 4).tolist()}", flush=True)
print(f"[smoke] SMOKE_OK total {time.time()-t0:.1f}s", flush=True)
