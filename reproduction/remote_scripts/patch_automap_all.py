import json, glob, os
d = "/root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial"
for p in glob.glob(os.path.join(d, "*.json")):
    with open(p) as f:
        s = f.read()
    if '"auto_map"' not in s or "--" not in s:
        continue
    obj = json.loads(s)
    changed = False
    for k, v in obj.get("auto_map", {}).items():
        if "--" in v:
            obj["auto_map"][k] = v.split("--", 1)[1]
            changed = True
    if changed:
        json.dump(obj, open(p, "w"), indent=2)
        print("patched", os.path.basename(p))
print("DONE")
