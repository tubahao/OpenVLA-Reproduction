import json
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

path = "/root/autodl-tmp/openvla/wandb/offline-run-20260812_224203-kpxz1sew/run-kpxz1sew.wandb"
ds = DataStore()
ds.open_for_scan(path)
rows = []
while True:
    try:
        r = ds.scan_record()
    except Exception:
        break
    if r is None:
        break
    dtype, data = r
    if dtype != 1:
        try:
            cont = ds.scan_data()
        except Exception:
            break
        if cont is None:
            break
        data = data + cont
    rec = pb.Record()
    try:
        rec.ParseFromString(data)
    except Exception:
        continue
    if rec.history and rec.history.item:
        vals = {}
        for item in rec.history.item:
            name = ".".join(item.nested_key) if item.nested_key else item.key
            if item.value_json:
                try:
                    vals[name] = json.loads(item.value_json)
                except Exception:
                    pass
        if vals:
            rows.append(vals)

print("ROWS", len(rows))
for r in rows[::200]:
    s = r.get("_step", "?")
    print("step", s, "| loss", round(r.get("train_loss", float("nan")), 3),
          "| acc", round(r.get("action_accuracy", float("nan")), 3),
          "| l1", round(r.get("l1_loss", float("nan")), 3))
if rows:
    r = rows[-1]
    print("FINAL step", r.get("_step"), "| loss", round(r.get("train_loss", float("nan")), 3),
          "| acc", round(r.get("action_accuracy", float("nan")), 3),
          "| l1", round(r.get("l1_loss", float("nan")), 3))
