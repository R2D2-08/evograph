#!/usr/bin/env python3
"""Copy existing local CAPE reports, without executing/submitting samples.

Labels are deliberately left blank: do not infer clean/malicious from the presence
of a CAPE report. You must establish labels independently and document provenance.
"""
import argparse,csv,json,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--cape-analyses',required=True,
   help='Path to CAPE storage/analyses directory')
 p.add_argument('--out',default='real_data');args=p.parse_args()
 src=Path(args.cape_analyses);out=Path(args.out)
 if not src.is_dir():raise SystemExit(f'Not a directory: {src}')
 reports=sorted(src.glob('*/reports/report.json'))
 (out/'reports').mkdir(parents=True,exist_ok=True); rows=[]
 for original in reports:
  task=original.parent.parent.name
  try:
   with original.open(encoding='utf-8') as fh:data=json.load(fh)
  except (json.JSONDecodeError,OSError):continue
  target=data.get('target') or {};file_info=target.get('file') or {}
  sha=file_info.get('sha256') or ''
  when=(data.get('info') or {}).get('started') or ''
  dest=out/'reports'/f'{task}.json'
  if not dest.exists():shutil.copy2(original,dest)
  rows.append([f'task_{task}',f'reports/{task}.json','',when,sha])
 with (out/'metadata.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['sample_id','report_path','label','observed_at','sha256']);w.writerows(rows)
 print(f'Copied/indexed {len(rows)} reports into {out}')
 print('NEXT: fill label (0=benign, 1=malware) and observed_at in metadata.csv.')
 print('LABELS ARE BLANK BY DESIGN. Exclude uncertain samples; never auto-label all reports as malware.')
if __name__=='__main__':main()
