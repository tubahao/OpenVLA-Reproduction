import sys
path = "/root/autodl-tmp/openvla/wandb/offline-run-20260812_224203-kpxz1sew/run-kpxz1sew.wandb"
try:
    from wandb.sdk.internal.datastore import DataStore
    ds = DataStore()
    ds.open_for_read(path)
    rows = []
    while ds.has_data():
        rec = ds.read_record()
        if rec.history and rec.history.item:
            vals = {}
            for item in rec.history.item:
                for f in ("int_value", "double_value", "float_value"):
                    if item.HasField(f):
                        vals[item.key] = getattr(item, f)
            if vals:
                rows.append(vals)
    print("ROWS", len(rows))
    for r in rows[:3]:
        print("FIRST", r)
    for r in rows[-3:]:
        print("LAST", r)
except Exception as e:
    print("ERR", type(e).__name__, e)
