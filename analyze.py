import pandas as pd
import numpy as np
import gc
import json
from collections import defaultdict

FILES = {
    '2022-05-23': ('Mon', 0, False),
    '2022-05-24': ('Tue', 1, False),
    '2022-05-25': ('Wed', 2, False),
    '2022-05-26': ('Thu', 3, False),
    '2022-05-27': ('Fri', 4, False),
    '2022-05-28': ('Sat', 5, True),
    '2022-05-29': ('Sun', 6, True),
}
RU_DOW = {'Mon':'Понедельник','Tue':'Вторник','Wed':'Среда','Thu':'Четверг','Fri':'Пятница','Sat':'Суббота','Sun':'Воскресенье'}

DATA_DIR = '/mnt/user-data/uploads'
USECOLS = ['timestamp', 'splituid', 'uuid', 'auth']
CHUNKSIZE = 300_000

# Per-day accumulators
daily_unique_splituid = {}          # date -> set of splituid (all traffic)
daily_unique_auth_uuid = {}         # date -> set of uuid (auth==1)
daily_events_total = {}
daily_events_auth = {}

# Global accumulators
global_splituid = set()
global_auth_uuid = set()
uuid_to_splituid = defaultdict(set)  # uuid -> set(splituid)  (Q3)

# Hourly matrices: rows=0..6 (date order), cols=0..23 ; separate weekday/weekend aggregation
hourly_events_auth = np.zeros((7, 24), dtype=np.int64)   # per-day-of-week x hour, event counts (auth users)
hourly_unique_auth_uuid_perday = {}  # date -> dict hour-> set(uuid)  (to get unique auth visitors per hour if needed)

for date, (dow, idx, is_weekend) in FILES.items():
    path = f'{DATA_DIR}/{date}.csv'
    print(f'Processing {date} ({dow})...')

    day_splituid_set = set()
    day_auth_uuid_set = set()
    day_events_total = 0
    day_events_auth = 0
    day_hour_uuid_sets = defaultdict(set)

    reader = pd.read_csv(path, usecols=USECOLS, dtype=str, chunksize=CHUNKSIZE)
    for chunk in reader:
        day_events_total += len(chunk)

        # splituid (cookie) — overall unique visitors
        sp = chunk['splituid'].dropna()
        day_splituid_set.update(sp.values)
        global_splituid.update(sp.values)

        # hour extraction (fixed format YYYY-MM-DDTHH:MM:SS...)
        hours = chunk['timestamp'].str.slice(11, 13)

        # authorized subset
        auth_mask = chunk['auth'] == '1'
        auth_chunk = chunk.loc[auth_mask]
        day_events_auth += len(auth_chunk)

        auth_uuid = auth_chunk['uuid'].dropna()
        day_auth_uuid_set.update(auth_uuid.values)
        global_auth_uuid.update(auth_uuid.values)

        # Q3: uuid -> set of splituid (cookies used by that authorized user)
        auth_sub = auth_chunk[['uuid', 'splituid']].dropna()
        if len(auth_sub):
            for u, s in zip(auth_sub['uuid'].values, auth_sub['splituid'].values):
                uuid_to_splituid[u].add(s)

        # Q4: hourly counts for authorized events
        auth_hours = hours.loc[auth_mask].dropna()
        if len(auth_hours):
            vc = auth_hours.value_counts()
            for h_str, cnt in vc.items():
                h = int(h_str)
                hourly_events_auth[idx, h] += cnt

        del chunk, sp, hours, auth_mask, auth_chunk, auth_uuid, auth_sub, auth_hours
    gc.collect()

    daily_unique_splituid[date] = len(day_splituid_set)
    daily_unique_auth_uuid[date] = len(day_auth_uuid_set)
    daily_events_total[date] = day_events_total
    daily_events_auth[date] = day_events_auth

    print(f'  events_total={day_events_total}, unique_splituid={len(day_splituid_set)}, '
          f'events_auth={day_events_auth}, unique_auth_uuid={len(day_auth_uuid_set)}')

    del day_splituid_set, day_auth_uuid_set
    gc.collect()

print('\n=== GLOBAL ===')
print('Total unique splituid (week):', len(global_splituid))
print('Total unique auth uuid (week):', len(global_auth_uuid))
print('Total unique authorized users with >=1 splituid mapping:', len(uuid_to_splituid))

# Save intermediate results
results = {
    'daily_unique_splituid': daily_unique_splituid,
    'daily_unique_auth_uuid': daily_unique_auth_uuid,
    'daily_events_total': daily_events_total,
    'daily_events_auth': daily_events_auth,
    'global_unique_splituid': len(global_splituid),
    'global_unique_auth_uuid': len(global_auth_uuid),
    'hourly_events_auth': hourly_events_auth.tolist(),
    'files_meta': FILES,
}

with open('/home/claude/work/results_part1.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Save uuid->cookie count distribution (Q3) - don't need full sets anymore, just counts
browser_counts = {u: len(s) for u, s in uuid_to_splituid.items()}
with open('/home/claude/work/browser_counts.json', 'w', encoding='utf-8') as f:
    json.dump(browser_counts, f)

print('\nSaved results_part1.json and browser_counts.json')
