import json
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

path = "/root/autodl-tmp/openvla/wandb/offline-run-20260812_224203-kpxz1sew/run-kpxz1sew.wandb"
ds = DataStore()
ds.open_for_scan(path)
rows = []
full = multi = 0
while True:
    try:
        r = ds.scan_record()
    except Exception as e:
        print("SCAN_STOP", type(e).__name__, e)
        break
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
            if item.value_json:
                try:
                    vals[item.key] = json.loads(item.value_json)
                except Exception:
                    pass
        if vals:
            rows.append(vals)
print("FULL", full, "MULTI", multi, "ROWS", len(rows))
for r in rows[:3]:
    print("FIRST", r)
for r in rows[-3:]:
    print("LAST", r)
