p = "/root/autodl-tmp/openvla/experiments/robot/openvla_utils.py"
s = open(p).read()
old = '        attn_implementation="flash_attention_2",\n'
new = '        # NOTE: flash_attention_2 removed (flash-attn not installed); eager attention used\n'
assert old in s, "pattern not found"
s = s.replace(old, new)
open(p, "w").write(s)
print("PATCHED_VLA_UTILS")
