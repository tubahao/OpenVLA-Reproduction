from wandb.sdk.internal.datastore import DataStore

path = "/root/autodl-tmp/openvla/wandb/offline-run-20260812_224203-kpxz1sew/run-kpxz1sew.wandb"
ds = DataStore()
ds.open_for_scan(path)
rows = []
while True:
    try:
        rec = ds.scan_record()
    except Exception as e:
        print("SCAN_ERR", e)
        break
    if rec is None:
        break
    if rec.history and rec.history.item:
        vals = {}
        for item in rec.history.item:
            for f in ("int_value", "double_value", "float_value"):
                if item.HasField(f):
                    vals[item.key] = getattr(item, f)
        if vals:
            rows.append(vals)
print("ROWS", len(rows))
for r in rows[:2]:
    print("FIRST", r)
for r in rows[-2:]:
    print("LAST", r)
