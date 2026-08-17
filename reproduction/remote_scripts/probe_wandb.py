import json, collections
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

path = "/root/autodl-tmp/openvla/wandb/offline-run-20260812_224203-kpxz1sew/run-kpxz1sew.wandb"
ds = DataStore()
ds.open_for_scan(path)
counts = collections.Counter()
n_history = 0
n_items = 0
item_keys = collections.Counter()
example_items = {}
while True:
    try:
        r = ds.scan_record()
    except Exception as e:
        print("SCAN_ERR", repr(e)); break
    if r is None: break
    dtype, data = r
    if dtype != 1:
        try:
            cont = ds.scan_data()
            if cont is None:
                print("EOF in chunked at offset", ds.get_offset()); break
            data = data + cont
        except Exception as e:
            print("SCAN_DATA_ERR", repr(e)); break
    rec = pb.Record()
    try:
        rec.ParseFromString(data)
    except Exception as e:
        counts["parse_err"] += 1
        continue
    counts[rec.WhichOneof("record_type")] += 1
    if rec.history and rec.history.item:
        n_history += 1
        n_items += len(rec.history.item)
        for item in rec.history.item:
            k = ".".join(item.nested_key) if item.nested_key else item.key
            item_keys[k] += 1
            if k in ("train_loss", "action_accuracy", "_step", "l1_loss") and k not in example_items:
                example_items[k] = item
print("COUNTS", dict(counts))
print("N_HISTORY", n_history, "N_ITEMS", n_items)
print("ITEM_KEYS", item_keys.most_common(30))
for k, item in example_items.items():
    print("EXAMPLE", k, "->", item)
