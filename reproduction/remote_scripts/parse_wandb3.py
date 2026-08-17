from wandb.proto import wandb_internal_pb2 as pb
from collections import Counter

path = "/root/autodl-tmp/openvla/wandb/offline-run-20260812_224203-kpxz1sew/run-kpxz1sew.wandb"
data = open(path, "rb").read()
offset = 12  # skip header
types = Counter()
rows = []
while offset < len(data):
    length = 0; shift = 0
    while True:
        b = data[offset]; offset += 1
        length |= (b & 0x7F) << shift
        if not (b & 0x80): break
        shift += 7
    payload = data[offset:offset + length]; offset += length
    rec = pb.Record()
    try:
        rec.ParseFromString(payload)
    except Exception:
        continue
    t = rec.WhichOneof("record_type")
    types[t] += 1
    if t == "history":
        vals = {}
        for item in rec.history.item:
            for field in ("int_value", "double_value", "float_value"):
                if item.HasField(field):
                    vals[item.key] = getattr(item, field)
        if vals:
            rows.append(vals)

print("TYPES", dict(types))
print("HISTORY_ROWS", len(rows))
if rows:
    for r in rows[:3]:
        print("FIRST", r)
    for r in rows[-3:]:
        print("LAST", r)
