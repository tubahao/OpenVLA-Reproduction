import json, sys
p = sys.argv[1]
d = json.load(open(p))
for k, v in d.get("auto_map", {}).items():
    if "--" in v:
        d["auto_map"][k] = v.split("--", 1)[1]
json.dump(d, open(p, "w"), indent=2)
print("patched auto_map:", d["auto_map"])
