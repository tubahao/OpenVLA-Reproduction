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
px = batch["pixel_values"]
print("pixel_values", tuple(px.shape), "mean", round(float(px.mean()), 3), "std", round(float(px.std()), 3), flush=True)
print("input_ids", tuple(batch["input_ids"].shape), "prompt_len", int((batch["input_ids"] != processor.tokenizer.pad_token_id).sum(1).max()), flush=True)
ids = batch["input_ids"][0]
text = processor.tokenizer.decode(ids[ids != processor.tokenizer.pad_token_id], skip_special_tokens=True)
print("PROMPT_TEXT", text[:120].replace("\n", " | "), flush=True)
lab = batch["action_labels"]
print("action_labels", tuple(lab.shape), "unique_tokens_first_dim", [len(np.unique(lab[:, i].numpy())) for i in range(min(7, lab.shape[1]))], flush=True)
acts = action_tokenizer.decode_token_ids_to_actions(lab.numpy())
print("raw_actions", np.round(acts, 3).tolist(), flush=True)
print("INSPECT_OK", flush=True)
