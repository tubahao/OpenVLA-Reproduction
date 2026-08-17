p = "/root/autodl-tmp/openvla/vla-scripts/finetune.py"
s = open(p).read()
anchor = "            # Push Metrics to W&B (every 10 gradient steps)"
addition = (
    "            # [DIAG] Print metrics to console every 50 gradient steps\n"
    "            if distributed_state.is_main_process and gradient_step_idx % 50 == 0:\n"
    "                print(f\"[DIAG step {gradient_step_idx}] loss={smoothened_loss:.3f} acc={smoothened_action_accuracy:.3f} l1={smoothened_l1_loss:.3f}\", flush=True)\n\n"
)
assert anchor in s, "anchor not found"
s = s.replace(anchor, addition + anchor, 1)
open(p, "w").write(s)
print("PATCHED_FINETUNE_PRINT")
