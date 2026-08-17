import os
import torch
import numpy as np
from transformers import AutoProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
VLA = "/root/autodl-tmp/checkpoints/openvla-7b"
DATA_ROOT = "/root/autodl-tmp/datasets/modified_libero_rlds"
DS_NAME = "libero_spatial_no_noops"

processor = AutoProcessor.from_pretrained(VLA, trust_remote_code=True)
action_tokenizer = ActionTokenizer(processor.tokenizer)
bt = RLDSBatchTransform(action_tokenizer, processor.tokenizer,
                        image_transform=processor.image_processor.apply_transform,
                        prompt_builder_fn=PurePromptBuilder)
ds = RLDSDataset(DATA_ROOT, DS_NAME, bt, resize_resolution=(224, 224),
                 shuffle_buffer_size=100, image_aug=True)
collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
                                             processor.tokenizer.pad_token_id, padding_side="right")
dl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collator, num_workers=0)
batch = next(iter(dl))
print("KEYS", list(batch.keys()), flush=True)
labels = batch["labels"]
input_ids = batch["input_ids"]
print("input_ids", tuple(input_ids.shape), "labels", tuple(labels.shape), flush=True)
lbl = labels[0]
ids = input_ids[0]
active = (lbl != -100)
act_pos = torch.nonzero(active).flatten()
print("active_label_positions", act_pos[:12].tolist(), "n_active", int(active.sum()), flush=True)
# causal check: labels[i] should equal input_ids[i+1] at active positions (except maybe final)
match = 0
total = 0
for i in act_pos:
    if i + 1 < ids.shape[0]:
        total += 1
        if lbl[i] == ids[i + 1]:
            match += 1
print("CAUSAL_ALIGN", match, "/", total, flush=True)
# token value ranges
print("label_value_range", int(lbl[active].min()), int(lbl[active].max()), flush=True)
print("action_token_begin_idx", action_tokenizer.action_token_begin_idx, flush=True)
print("INSPECT_LABELS_OK", flush=True)
