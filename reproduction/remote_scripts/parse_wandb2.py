from wandb.proto import wandb_internal_pb2 as pb
from collections import Counter

path = "/root/autodl-tmp/openvla/wandb/offline-run-20260812_224203-kpxz1sew/run-kpxz1sew.wandb"
data = open(path, "rb").read()
offset = 0
types = Counter()
parsed = 0
n_records = 0
while offset < len(data):
    length = 0; shift = 0
    while True:
        b = data[offset]; offset += 1
        length |= (b & 0x7F) << shift
        if not (b & 0x80): break
        shift += 7
    payload = data[offset:offset + length]; offset += length
    n_records += 1
    rec = pb.Record()
    try:
        rec.ParseFromString(payload)
        parsed += 1
        t = rec.WhichOneof("record_type")
        types[t] += 1
        if t == "history" and types[t] <= 2:
            print("HISTORY_FIELDS", rec.history)
            print("HISTORY_ITEM_LEN", len(rec.history.item))
    except Exception as e:
        pass
print("N_RECORDS", n_records, "PARSED", parsed)
print("TYPES", dict(types))
