from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb
path = "/root/autodl-tmp/openvla/wandb/offline-run-20260812_224203-kpxz1sew/run-kpxz1sew.wandb"
ds = DataStore(); ds.open_for_scan(path)
while True:
    try:
        data = ds.scan_data()
    except Exception:
        break
    if data is None: break
    rec = pb.Record()
    try: rec.ParseFromString(data)
    except Exception: continue
    if rec.HasField("run"):
        print(rec.run.config)
        break
