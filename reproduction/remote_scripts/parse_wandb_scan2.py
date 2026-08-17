from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

path = "/root/autodl-tmp/openvla/wandb/offline-run-20260812_224203-kpxz1sew/run-kpxz1sew.wandb"
ds = DataStore()
ds.open_for_scan(path)
rows = []
full = multi = 0
while True:
    r = ds.scan_record()
    if r is None:
        break
    dtype, data = r
    if dtype != 1:
        multi += 1
        continue
    full += 1
    rec = pb.Record()
    try:
        rec.ParseFromString(data)
    except Exception:
        continue
    if rec.history and rec.history.item:
        vals = {}
        for item in rec.history.item:
            for f in ("int_value", "double_value", "float_value"):
                if item.HasField(f):
                    vals[item.key] = getattr(item, f)
        if vals:
            rows.append(vals)
print("FULL", full, "MULTI", multi, "ROWS", len(rows))
for r in rows[:2]:
    print("FIRST", r)
for r in rows[-2:]:
    print("LAST", r)
