import csv, os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'metadata', 'dataset_metadata.csv')
NEW_PREFIX = os.path.join('data', 'raw', 'asvspoof_voice')
OLD_PREFIX = r'C:\Users\user\Downloads\_sources\_sources'

missing = []
seen = set()
rows = []
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        old_path = row['file_path']
        # replace prefix
        if old_path.startswith(OLD_PREFIX):
            rel_path = os.path.relpath(old_path, OLD_PREFIX)
            new_path = os.path.normpath(os.path.join(PROJECT_ROOT, NEW_PREFIX, rel_path))
            # store relative path (relative to project root) as used by loader
            row['file_path'] = new_path
        else:
            # keep unchanged (perhaps already relative)
            row['file_path'] = os.path.normpath(os.path.join(PROJECT_ROOT, old_path))
        # track duplicates
        if row['file_path'] in seen:
            duplicate = row['file_path']
        else:
            seen.add(row['file_path'])
        # verify existence
        if not os.path.isfile(row['file_path']):
            missing.append(row['file_path'])
        rows.append(row)

# write back CSV (preserve header order)
fieldnames = rows[0].keys() if rows else []
with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print('Total rows:', len(rows))
print('Missing files:', len(missing))
if missing:
    for m in missing[:20]:
        print('MISSING:', m)
print('Duplicate paths detected:', len(rows) - len(seen))
# sanity check class counts
real = sum(1 for r in rows if r['label'] == 'real')
fake = sum(1 for r in rows if r['label'] == 'fake')
bg = sum(1 for r in rows if r['label'] == 'background_noise')
print('Counts - real:', real, 'fake:', fake, 'background:', bg)
