#!/usr/bin/env python3
"""Minimal reproducible CAPE behavior-graph detector. Research/demo only."""
import argparse, copy, csv, hashlib, json, math, random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
import torch
from torch import nn
from torch.nn import functional as F

# Stable, explicit, deliberately small capability mapping; not a malware-label oracle.
CAPABILITIES = [
    'file_io','registry','process','memory','network','service','crypto',
    'system_info','thread','other'
]
API_MAP = {
 'file_io': ['createfile','ntcreatefile','writefile','ntwritefile','readfile','ntreadfile','deletefile','copyfile','movefile','getfileattributes','setfileattributes'],
 'registry': ['regsetvalue','regsetvalueex','regcreatekey','regopenkey','regdeletekey','regqueryvalue','ntsetvaluekey','ntcreatekey','ntopenkey'],
 'process': ['createprocess','createprocessinternal','ntcreateuserprocess','shellexecute','winexec','openprocess','terminateprocess','getprocessid'],
 'memory': ['virtualalloc','virtualallocex','virtualprotect','virtualprotectex','writeprocessmemory','readprocessmemory','ntallocatevirtualmemory','ntwritevirtualmemory'],
 'network': ['connect','wsaconnect','wsasend','wsarecv','send','recv','internetconnect','httpsendrequest','winhttpconnect','winhttpsendrequest','dnsquery','getaddrinfo','urldownloadtofile'],
 'service': ['createservice','startservice','openservice','controlservice','deleteservice'],
 'crypto': ['cryptencrypt','cryptdecrypt','bcryptencrypt','bcryptdecrypt','cryptacquirecontext','cryptgenkey'],
 'system_info': ['getcomputername','getusername','getversion','getsysteminfo','gettickcount','getadaptersinfo'],
 'thread': ['createthread','createremotethread','ntcreatethreadex','resumethread','suspendthread']
}
API_MAP = {k: [v.replace(' ', '') for v in values] for k, values in API_MAP.items()}
NODE_TYPES = ['process','api','capability']
HASH_BUCKETS = 32
DIM = 3 + len(CAPABILITIES) + HASH_BUCKETS + 1


def capability(name):
    base = str(name).lower().split('!')[-1].split('.')[-1]
    for cap, names in API_MAP.items():
        if any(base.startswith(p) for p in names): return cap
    return 'other'


def parse_report(path, max_processes=16, max_apis=120):
    """Reads behavior.processes[].calls and falls back to behavior.apistats. No labels inferred."""
    with open(path, encoding='utf-8') as fh: doc = json.load(fh)
    behavior = doc.get('behavior') or {}
    processes = []
    for i, proc in enumerate((behavior.get('processes') or [])[:max_processes]):
        counts = Counter()
        calls = proc.get('calls') or []
        if isinstance(calls, list):
            for c in calls:
                if isinstance(c, dict) and c.get('api'):
                    counts[str(c['api']).strip().lower()] += 1
        if counts:
            processes.append({'name': f'process_{i}', 'calls': dict(counts.most_common(max_apis))})
    if not processes:
        # Some CAPE configurations store API counts without the complete ordered call trace.
        stats = behavior.get('apistats') or {}
        if isinstance(stats, dict):
            for i, (_pid, values) in enumerate(list(stats.items())[:max_processes]):
                if isinstance(values, dict):
                    cnt = {str(k).strip().lower():int(v) for k,v in values.items()
                           if str(k).strip() and isinstance(v,(int,float)) and v>0}
                    if cnt: processes.append({'name':f'process_{i}', 'calls': dict(Counter(cnt).most_common(max_apis))})
    if not processes:
        # Some reduced CAPE configurations preserve resolved API names in the summary only.
        summary = behavior.get('summary') or {}
        resolved = summary.get('resolved_apis') or []
        if isinstance(resolved, list) and resolved:
            counts = Counter(str(x).strip().lower() for x in resolved if str(x).strip())
            if counts:
                processes = [{'name': 'summary_process', 'calls': dict(counts.most_common(max_apis))}]
    if not processes:
        raise ValueError('No API events found under behavior.processes[].calls, behavior.apistats, or behavior.summary.resolved_apis')
    return processes


def labeled_rows(metadata):
    rows = []
    with open(metadata, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        needed = {'sample_id','report_path','label','observed_at'}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError('metadata.csv requires: sample_id,report_path,label,observed_at,sha256')
        for r in reader:
            label = str(r['label']).strip().lower()
            if not label: continue  # Unlabeled input intentionally not used.
            if label not in ('0','1','benign','malware'):
                raise ValueError(f"Invalid label for {r['sample_id']}: {label!r}")
            r['label'] = 1 if label in ('1','malware') else 0
            if not r['sample_id'].strip(): raise ValueError('Missing sample_id')
            from datetime import datetime, timezone
            stamp = str(r.get('observed_at') or '').strip()
            if stamp:
                try:
                    dt = datetime.fromisoformat(stamp.replace('Z','+00:00'))
                    if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
                    r['observed_at'] = dt.astimezone(timezone.utc).isoformat()
                except ValueError as ex:
                    raise ValueError(f'Invalid ISO observed_at for {r["sample_id"]}: {stamp}') from ex
            else:
                # Missing execution timestamps remain missing; never fabricate chronology.
                r['observed_at'] = ''
            raw_path = Path(r['report_path'])
            if not raw_path.is_absolute(): raw_path = (Path(metadata).resolve().parent / raw_path)
            r['report_path'] = str(raw_path.resolve())
            rows.append(r)
    return rows


def prepare(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=labeled_rows(args.metadata)
    collected=[]; seen=set(); skipped=[]
    for row in rows:
        key = (row.get('sha256') or '').strip().lower() or row['sample_id']
        if key in seen:
            skipped.append((row['sample_id'],'duplicate sha256/sample_id')); continue
        seen.add(key)
        try: procs=parse_report(row['report_path'])
        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as ex:
            skipped.append((row['sample_id'],str(ex))); continue
        collected.append({k:row.get(k,'') for k in ('sample_id','observed_at','sha256','report_path')}
                         | {'label':row['label'],'processes':procs})
    collected.sort(key=lambda r:(r['observed_at'],r['sample_id']))
    path=out/'dataset.jsonl'
    with path.open('w',encoding='utf-8') as fh:
        for r in collected: fh.write(json.dumps(r)+'\n')
    with (out/'skipped.csv').open('w',newline='') as fh:
        w=csv.writer(fh); w.writerow(['sample_id','reason']); w.writerows(skipped)
    labels=Counter(x['label'] for x in collected)
    print(f'Prepared {len(collected)} reports (benign={labels[0]}, malware={labels[1]}), skipped={len(skipped)} -> {path}')
    if len(collected)<24: print('WARNING: small dataset; training needs >=24 with both labels in every chronological window.')
    if skipped: print('See skipped.csv for reports with missing traces, duplicates, or missing files.')


def load_dataset(path):
    with open(path,encoding='utf-8') as f: data=[json.loads(line) for line in f if line.strip()]
    if len(data)<24: raise ValueError('Need at least 24 labeled, parseable reports for this illustrative experiment.')
    return sorted(data,key=lambda r:(r.get('observed_at') or '',r['sample_id']))


def make_splits(data, mode='stratified', seed=42):
    n=len(data)
    y=np.asarray([int(d['label']) for d in data])
    if len(set(y.tolist())) < 2:
        raise ValueError('Both benign and malware labels are required.')
    if mode == 'chronological':
        if any(not d.get('observed_at') for d in data):
            raise ValueError('Chronological split requires a genuine CAPE execution timestamp for every report. Use --split-mode stratified otherwise.')
        stops=[int(n*.50),int(n*.65),int(n*.80),n]
        a,b,c=stops[:3]
        groups={'initial_train':list(range(0,a)),'validation':list(range(a,b)),
                'update_batch':list(range(b,c)),'heldout_test':list(range(c,n))}
        for earlier,later in [('initial_train','validation'),('validation','update_batch'),('update_batch','heldout_test')]:
            if data[groups[earlier][-1]]['observed_at'] >= data[groups[later][0]]['observed_at']:
                raise ValueError('Chronological split boundary has tied/overlapping observation times; collect data across distinct observation periods.')
    elif mode == 'stratified':
        idx=np.arange(n)
        dev, test = train_test_split(idx, test_size=.20, random_state=seed, stratify=y)
        train, val_update = train_test_split(dev, test_size=.375, random_state=seed+1, stratify=y[dev])
        val, update = train_test_split(val_update, test_size=.50, random_state=seed+2, stratify=y[val_update])
        groups={'initial_train':train.tolist(),'validation':val.tolist(),
                'update_batch':update.tolist(),'heldout_test':test.tolist()}
    else:
        raise ValueError(f'Unknown split mode: {mode}')
    for name,idx in groups.items():
        labels={data[i]['label'] for i in idx}
        if len(labels)<2:
            raise ValueError(f'{name} has only one class. Increase the dataset or adjust the split.')
    return groups


def feature_dict(item):
    vec=Counter()
    for p in item['processes']:
        for api, count in p['calls'].items():
            vec['api:'+api] += count
    vec['n_processes']=len(item['processes'])
    return {k:math.log1p(v) for k,v in vec.items()}


def graph(item):
    """Graph with process -> API -> capability, plus reverse edges for message passing."""
    nodes=[]; edges=[]; keys={}
    def add(key, typ, name, count=1):
        if key not in keys:
            keys[key]=len(nodes); nodes.append((typ,name,count))
        return keys[key]
    for pi, proc in enumerate(item['processes']):
        p=add(('p',pi),'process',f'process_{pi}')
        for api, ct in proc['calls'].items():
            api_node=add(('a',pi,api),'api',api,ct)
            cap=capability(api)
            cap_node=add(('c',cap),'capability',cap)
            edges += [(p,api_node),(api_node,p),(api_node,cap_node),(cap_node,api_node)]
    x=np.zeros((len(nodes),DIM),dtype=np.float32)
    for i,(typ,name,count) in enumerate(nodes):
        x[i,NODE_TYPES.index(typ)]=1
        if typ=='api':
            bucket=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'big')%HASH_BUCKETS
            x[i,3+len(CAPABILITIES)+bucket]=1
            x[i,-1]=math.log1p(count)/8
            x[i,3+CAPABILITIES.index(capability(name))]=1
        elif typ=='capability': x[i,3+CAPABILITIES.index(name)]=1
    edge=np.array(edges,dtype=np.int64).T if edges else np.zeros((2,0),dtype=np.int64)
    return x,edge,nodes,edges


def batches(data,indices,batch_size=16,shuffle=False):
    ix=list(indices)
    if shuffle: random.shuffle(ix)
    for offset in range(0,len(ix),batch_size):
        selected=ix[offset:offset+batch_size]
        xx=[];ee=[];bb=[];ys=[];start=0
        for bi,i in enumerate(selected):
            x,e,_,_=graph(data[i]); xx.append(torch.from_numpy(x))
            ee.append(torch.from_numpy(e)+start)
            bb.append(torch.full((len(x),),bi,dtype=torch.long))
            ys.append(data[i]['label']);start+=len(x)
        yield (torch.cat(xx),torch.cat(ee,dim=1),torch.cat(bb),
               torch.tensor(ys,dtype=torch.float32),selected)


class MeanGraphLayer(nn.Module):
    def __init__(self, inp, out):
        super().__init__();self.selflin=nn.Linear(inp,out);self.neighlin=nn.Linear(inp,out)
    def forward(self,x,edge):
        src,dst=edge
        sums=torch.zeros_like(x)
        sums.index_add_(0,dst,x[src])
        count=torch.zeros((len(x),1),dtype=x.dtype,device=x.device)
        count.index_add_(0,dst,torch.ones((len(dst),1),device=x.device))
        return F.relu(self.selflin(x)+self.neighlin(sums/count.clamp(min=1)))


class GraphClassifier(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.l1=MeanGraphLayer(DIM,hidden);self.l2=MeanGraphLayer(hidden,hidden)
        self.head=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Dropout(.15),nn.Linear(hidden,1))
    def forward(self,x,edge,batch):
        h=self.l1(x,edge);h=F.dropout(h,p=.10,training=self.training)
        h=self.l2(h,edge)
        b=int(batch.max().item())+1
        pooled=torch.zeros((b,h.shape[1]),device=h.device)
        pooled.index_add_(0,batch,h)
        counts=torch.bincount(batch,minlength=b).float().unsqueeze(1).clamp(min=1)
        pooled_mean=pooled/counts
        pooled_sqrt=pooled/counts.sqrt()  # retains some graph-size information
        return self.head(torch.cat([pooled_mean,pooled_sqrt],dim=1)).flatten()


def train_graph(model,data,train_idx,val_idx,epochs,lr,seed,patience=8):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.001)
    ys=[data[i]['label'] for i in train_idx]
    pos=sum(ys);neg=len(ys)-pos
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/max(pos,1)],dtype=torch.float32))
    best=1e30;best_state=copy.deepcopy(model.state_dict());since=0
    for ep in range(epochs):
        model.train()
        for x,e,b,y,_ in batches(data,train_idx,shuffle=True):
            opt.zero_grad();loss=criterion(model(x,e,b),y);loss.backward();opt.step()
        model.eval(); losses=[]
        with torch.no_grad():
            for x,e,b,y,_ in batches(data,val_idx):
                losses.append((criterion(model(x,e,b),y).item(),len(y)))
        v=sum(l*n for l,n in losses)/sum(n for _,n in losses)
        if v<best-0.0001:best=v;since=0;best_state=copy.deepcopy(model.state_dict())
        else:
            since+=1
            if since>=patience: break
    model.load_state_dict(best_state)
    return ep+1,best


def predict_graph(model,data,indices):
    model.eval();res=[]
    with torch.no_grad():
        for x,e,b,_y,_ in batches(data,indices):
            res.extend(torch.sigmoid(model(x,e,b)).cpu().tolist())
    return np.asarray(res)


def pick_threshold(y,prob):
    return float(max(np.arange(.30,.701,.05),key=lambda t:f1_score(y,prob>=t,zero_division=0)))


def metrics(y,proba,t):
    pred=(np.asarray(proba)>=t).astype(int)
    return {'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'accuracy':float(accuracy_score(y,pred)),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
            'n':len(y),'threshold':float(t)}


def render_sample_graph(item,path):
    _x,_edge,nodes,edges=graph(item)
    subset=min(len(nodes),32)
    g=nx.DiGraph();g.add_nodes_from(range(subset))
    g.add_edges_from((u,v) for u,v in edges if u<subset and v<subset and u<v)
    labels={i:(nodes[i][1][:19]) for i in range(subset)}
    colors={'process':'#3973c1','api':'#21a09a','capability':'#9b65cf'}
    plt.figure(figsize=(11,7))
    pos=nx.spring_layout(g,seed=12)
    nx.draw_networkx(g,pos,labels=labels,node_size=950,font_size=7,arrows=True,
                     node_color=[colors[nodes[i][0]] for i in g.nodes],edge_color='#888888')
    plt.title('Observed CAPE process → API → capability graph (truncated view)')
    plt.axis('off');plt.tight_layout();plt.savefig(path,dpi=150);plt.close()


def save_plot(results,out,split_mode):
    names=['Random Forest','Graph (before)','Graph (after)']
    values=[results[n]['f1'] for n in ['random_forest','graph_before','graph_after']]
    plt.figure(figsize=(8,4)); bars=plt.bar(names,values)
    for bar,v in zip(bars,values): plt.text(bar.get_x()+bar.get_width()/2,v+.012,f'{v:.2f}',ha='center')
    plt.ylim(0,1.10);plt.ylabel('F1 on held-out test');plt.title('Chronological holdout' if split_mode=='chronological' else 'Stratified holdout (not drift evaluation)')
    plt.tight_layout();plt.savefig(out/'f1_comparison.png',dpi=150);plt.close()
    cm=np.asarray(results['graph_after']['confusion_matrix'])
    plt.figure(figsize=(5,4));plt.imshow(cm,cmap='Blues')
    plt.xticks([0,1],['Benign','Malware']);plt.yticks([0,1],['Benign','Malware'])
    plt.xlabel('Predicted');plt.ylabel('True');plt.title('Updated graph model: future test')
    for i in range(2):
        for j in range(2):plt.text(j,i,str(cm[i,j]),ha='center',va='center',color='black')
    plt.tight_layout();plt.savefig(out/'confusion_matrix.png',dpi=150);plt.close()


def train(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data=load_dataset(args.dataset);groups=make_splits(data,args.split_mode,args.seed)
    with (out/'splits.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','observed_at','label','split','split_mode'])
        for name,inds in groups.items():
            for i in inds:w.writerow([data[i]['sample_id'],data[i].get('observed_at',''),data[i]['label'],name,args.split_mode])
    tr=groups['initial_train'];val=groups['validation'];up=groups['update_batch'];test=groups['heldout_test']
    dicts=[feature_dict(x) for x in data]
    vec=DictVectorizer(sparse=True)
    Xtr=vec.fit_transform([dicts[i] for i in tr])
    rf=RandomForestClassifier(n_estimators=150,class_weight='balanced_subsample',random_state=args.seed,n_jobs=2)
    rf.fit(Xtr,[data[i]['label'] for i in tr])
    def rfprobs(ix):return rf.predict_proba(vec.transform([dicts[i] for i in ix]))[:,1]
    rft=pick_threshold([data[i]['label'] for i in val],rfprobs(val))
    ytest=[data[i]['label'] for i in test]
    rf_results=metrics(ytest,rfprobs(test),rft)
    joblib.dump({'vectorizer':vec,'classifier':rf,'threshold':rft},out/'random_forest.joblib')
    model=GraphClassifier(args.hidden)
    old_epochs,old_loss=train_graph(model,data,tr,val,args.epochs,args.lr,args.seed)
    gt=pick_threshold([data[i]['label'] for i in val],predict_graph(model,data,val))
    before_probs=predict_graph(model,data,test)
    before=metrics(ytest,before_probs,gt)
    torch.save({'state_dict':model.state_dict(),'hidden':args.hidden,'threshold':gt,'schema':'cape-minimal-v1'},out/'graph_before.pt')
    # For continual update, retain a representative subset of old samples as replay.
    # Class-balanced oldest-memory replay (no use of future-test labels).
    by_label={c:[i for i in tr if data[i]['label']==c] for c in (0,1)}
    replay=[]
    for c in (0,1):
        random.shuffle(by_label[c]);replay+=by_label[c][:min(len(by_label[c]),args.replay//2)]
    update_idx=up+replay
    updated_epochs,update_loss=train_graph(model,data,update_idx,val,args.update_epochs,args.update_lr,args.seed,patience=5)
    after_probs=predict_graph(model,data,test)
    after=metrics(ytest,after_probs,gt)
    torch.save({'state_dict':model.state_dict(),'hidden':args.hidden,'threshold':gt,'schema':'cape-minimal-v1'},out/'graph_after.pt')
    result={'sample_count':len(data),'split_mode':args.split_mode,'splits':{k:len(v) for k,v in groups.items()},
            'random_forest':rf_results,'graph_before':before,'graph_after':after,
            'train_epochs':old_epochs,'update_epochs':updated_epochs,
            'replay_count':len(replay),
            'limitations':['Small dataset is a demonstration, not a validated deployment detector.',
                           ('This tests one chronological update step, not automatic drift adaptation.' if args.split_mode=='chronological' else 'This is a stratified incremental-update demonstration, NOT a temporal concept-drift evaluation.'),
                           'Behavioral nodes are simplified and do not capture event order.',
                           'Future test must remain untouched during model and threshold selection.']}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','observed_at','true_label','rf_probability','gnn_before_probability','gnn_after_probability','split_mode'])
        for j,i in enumerate(test):
            w.writerow([data[i]['sample_id'],data[i]['observed_at'],data[i]['label'],
                        round(float(rfprobs([i])[0]),5),round(float(before_probs[j]),5),round(float(after_probs[j]),5),args.split_mode])
    save_plot(result,out,args.split_mode);render_sample_graph(data[tr[0]],out/'sample_graph.png')
    print(json.dumps({k:v for k,v in result.items() if k not in ('limitations',)},indent=2))
    print('Results:',out.resolve())
    print('IMPORTANT: Report the split mode. Stratified results do not demonstrate concept drift.')


def predict(args):
    procs=parse_report(args.report)
    item={'sample_id':'unlabeled_input','label':0,'processes':procs}
    checkpoint=torch.load(Path(args.model_dir)/'graph_after.pt',map_location='cpu',weights_only=True)
    if checkpoint['schema']!='cape-minimal-v1':raise ValueError('Unsupported feature schema')
    model=GraphClassifier(checkpoint['hidden']);model.load_state_dict(checkpoint['state_dict'])
    p=predict_graph(model,[item],[0])[0]; t=checkpoint['threshold']
    print(json.dumps({'malware_probability':round(float(p),4),'threshold':t,
                      'predicted_label':'malware' if p>=t else 'benign',
                      'warning':'Research prototype; not a production safety verdict.'},indent=2))


def inspect(args):
    p=parse_report(args.report)
    apis=Counter()
    for proc in p:apis.update(proc['calls'])
    print(json.dumps({'processes':len(p),'total_api_calls':sum(apis.values()),
                      'unique_apis':len(apis),'top_apis':apis.most_common(12),
                      'capabilities':dict(Counter({cap:sum(c for a,c in apis.items() if capability(a)==cap)
                          for cap in CAPABILITIES}))},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    i=sub.add_parser('inspect');i.add_argument('--report',required=True);i.set_defaults(func=inspect)
    b=sub.add_parser('prepare');b.add_argument('--metadata',required=True);b.add_argument('--out',default='prepared');b.set_defaults(func=prepare)
    t=sub.add_parser('train');t.add_argument('--dataset',default='prepared/dataset.jsonl');t.add_argument('--out',default='results')
    t.add_argument('--epochs',type=int,default=24);t.add_argument('--update-epochs',type=int,default=12)
    t.add_argument('--lr',type=float,default=.004);t.add_argument('--update-lr',type=float,default=.001)
    t.add_argument('--hidden',type=int,default=48);t.add_argument('--replay',type=int,default=24)
    t.add_argument('--seed',type=int,default=42);t.add_argument('--split-mode',choices=['stratified','chronological'],default='stratified');t.set_defaults(func=train)
    r=sub.add_parser('predict');r.add_argument('--report',required=True);r.add_argument('--model-dir',default='results');r.set_defaults(func=predict)
    args=p.parse_args();args.func(args)

if __name__=='__main__':main()
