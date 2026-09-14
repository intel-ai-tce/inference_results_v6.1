import os, json
from datasets import load_dataset

CAL_LIST = "/cal/calibration-list.txt"
OUT_DIR = "/work/mlperf_l31_calib"
os.makedirs(OUT_DIR, exist_ok=True)

ids = [l.strip() for l in open(CAL_LIST) if l.strip()]
print("official_ids", len(ids))
idset = set(ids)

ds = load_dataset("abisee/cnn_dailymail", "3.0.0")
found = {}          # id -> article
split_hits = {}
for sp in ["validation", "test", "train"]:
    d = ds[sp]
    for i, a in zip(d["id"], d["article"]):
        if i in idset and i not in found:
            found[i] = a
            split_hits[sp] = split_hits.get(sp, 0) + 1
    # early exit if all found
    if len(found) == len(idset):
        break

print("matched", len(found), "of", len(ids))
print("split_distribution", split_hits)
missing = [x for x in ids if x not in found]
print("missing_count", len(missing), "first_missing", missing[:3])

# Write in official-list order, skipping any missing, using the 'text' column
# (modelopt os.path.isdir branch reads dataset['text']).
with open(os.path.join(OUT_DIR, "train.jsonl"), "w") as f:
    n = 0
    for x in ids:
        if x in found:
            f.write(json.dumps({"text": found[x], "id": x}) + "\n")
            n += 1
print("wrote", n, "rows ->", os.path.join(OUT_DIR, "train.jsonl"))
