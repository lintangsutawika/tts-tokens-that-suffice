import json, os, glob, collections
job = os.path.expanduser("~/tts-tokens-that-suffice/jobs/verified-Qwen--Qwen3.8-27B-FP8-sum-model-sectioned-48k-run-0")
c = collections.Counter()
n = 0
for rj in glob.glob(os.path.join(job, "*/result.json")):
    n += 1
    try:
        d = json.load(open(rj))
        ei = d.get("exception_info")
        c[ei.get("exception_type") if ei else "OK"] += 1
    except Exception:
        c["unreadable"] += 1
print("total result.json:", n)
for k, v in c.most_common():
    print("  ", k, v)

lk = json.load(open(os.path.join(job, "lock.json")))
locktrials = {t["name"] for t in lk["trials"]}
dirs = {os.path.basename(d) for d in glob.glob(os.path.join(job, "*/"))}
print("lock trials:", len(locktrials), " dirs:", len(dirs))
print("in lock not as dir:", len(locktrials - dirs))
print("dirs not in lock:", len(dirs - locktrials))