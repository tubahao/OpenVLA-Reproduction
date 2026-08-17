import os
import torch
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
batch_transform = RLDSBatchTransform(
    action_tokenizer,
    processor.tokenizer,
    image_transform=processor.image_processor.apply_transform,
    prompt_builder_fn=PurePromptBuilder,
)
dataset = RLDSDataset(
    DATA_ROOT,
    DS_NAME,
    batch_transform,
    resize_resolution=(224, 224),
    shuffle_buffer_size=100,
    image_aug=True,
)
print("NUM_EPISODES", len(dataset), flush=True)
collator = PaddedCollatorForActionPrediction(
    processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
)
dl = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collator, num_workers=0)
for i, batch in enumerate(dl):
    shapes = {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")}
    print("BATCH", i, shapes, flush=True)
    if i >= 1:
        break
print("DRYRUN_OK", flush=True)
