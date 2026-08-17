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
ids = batch["input_ids"][0].numpy()
lbl = batch["labels"][0].numpy()
print("SEQ_LEN", len(ids), flush=True)
for i in range(len(ids) - 12, len(ids)):
    print(i, "in:", ids[i], "lab:", lbl[i], flush=True)
print("self_align", int((lbl[ids != processor.tokenizer.pad_token_id] == ids[ids != processor.tokenizer.pad_token_id]).sum()), "/", int((ids != processor.tokenizer.pad_token_id).sum()), flush=True)
# check the actual model loss behavior quickly
import torch as T
print("SHOW_OK", flush=True)
