import csv
from datetime import datetime

with open("presence_log.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

present_times = []

for r in rows:
    if r["Status"]=="Present":
        present_times.append(datetime.strptime(r["Time"],"%H:%M:%S"))

if len(present_times)>=2:
    duration = (present_times[-1]-present_times[0]).seconds
    print("Total Presence:",duration,"seconds")
else:
    print("Not enough data")