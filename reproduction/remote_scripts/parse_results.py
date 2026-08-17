import re
p = "/root/autodl-tmp/openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_08_07-20_00_07.txt"
rates = []
cur = None
for line in open(p):
    line = line.strip()
    m = re.match(r"Task: (.+)", line)
    if m:
        cur = m.group(1)
    m2 = re.match(r"Current task success rate: ([\d.]+)", line)
    if m2 and cur:
        rates.append((cur, float(m2.group(1))))
        cur = None
for i, (t, r) in enumerate(rates, 1):
    print(f"{i:2d}. {r*100:5.1f}%  {t}")
print("TOTAL", round(sum(r for _, r in rates)/len(rates)*100, 1))
