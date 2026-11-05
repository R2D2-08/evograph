#!/usr/bin/env python3
"""Create FAKE CAPE-shaped reports to test the code only; NOT malware research data."""
import argparse,csv,hashlib,json,random
from datetime import datetime,timedelta,timezone
from pathlib import Path

BENIGN=['CreateFileW','ReadFile','GetComputerNameW','GetVersionExW','GetUserNameW','RegOpenKeyExW','GetSystemInfo','GetTickCount','CloseHandle','GetFileAttributesW']
MAL=['VirtualAllocEx','WriteProcessMemory','CreateRemoteThread','RegSetValueExW','CreateProcessW','connect','send','WSAConnect','WSASend','CreateFileW','CryptEncrypt']
BENIGN_LATE=['NtCreateFile','NtReadFile','GetComputerNameW','GetSystemInfo','RegQueryValueExW','GetTickCount','CloseHandle']
MAL_LATE=['NtAllocateVirtualMemory','NtWriteVirtualMemory','NtCreateThreadEx','NtSetValueKey','NtCreateUserProcess','WSAConnect','WSASend','NtCreateFile','CryptEncrypt']


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='synthetic_demo');p.add_argument('--n',type=int,default=120);args=p.parse_args()
 if args.n<40:raise ValueError('Use at least 40 generated examples')
 out=Path(args.out);(out/'reports').mkdir(parents=True,exist_ok=True)
 r=random.Random(123);rows=[];base=datetime(2025,1,1,tzinfo=timezone.utc)
 for i in range(args.n):
  label=i%2
  late=i>=int(args.n*.65)
  pool=(MAL_LATE if late else MAL) if label else (BENIGN_LATE if late else BENIGN)
  api_calls=r.choices(pool,k=r.randint(18,38))
  # modest overlap so classes are not perfectly separated
  api_calls.extend(r.choices(BENIGN+MAL,k=4))
  proc={'process_id':1200+i,'process_name':'example.exe',
        'calls':[{'api':api,'timestamp':float(j)*.01} for j,api in enumerate(api_calls)]}
  doc={'info':{'started':(base+timedelta(days=i)).isoformat()},'behavior':{'processes':[proc]},
       'info_note':'SYNTHETIC TEST DATA ONLY; not a real CAPE execution'}
  sid=f'fake_{i:04d}';raw=json.dumps(doc,indent=2)
  (out/'reports'/f'{sid}.json').write_text(raw,encoding='utf-8')
  rows.append([sid,f'reports/{sid}.json',label,(base+timedelta(days=i)).isoformat(),
               hashlib.sha256(f'synthetic-{i}'.encode()).hexdigest()])
 with (out/'metadata.csv').open('w',newline='') as fh:
  w=csv.writer(fh);w.writerow(['sample_id','report_path','label','observed_at','sha256']);w.writerows(rows)
 print(f'Wrote {args.n} SYNTHETIC reports under {out}/ (not scientific results)')
if __name__=='__main__':main()
