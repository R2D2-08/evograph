import json, tempfile, unittest
from pathlib import Path
import numpy as np
from run import parse_report, graph, capability, feature_dict, make_splits

class TestMinimalPipeline(unittest.TestCase):
    def test_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'report.json'
            p.write_text(json.dumps({'behavior':{'processes':[{'process_id':1,'calls':[{'api':'CreateFileW'},{'api':'CreateFileW'},{'api':'RegSetValueExW'}]}]}}))
            procs=parse_report(p)
            self.assertEqual(procs[0]['calls']['createfilew'],2)
            item={'processes':procs}
            x,edge,nodes,_=graph(item)
            self.assertEqual(x.shape[1],46)
            self.assertGreater(edge.shape[1],0)
            self.assertEqual(capability('RegSetValueExW'),'registry')
            self.assertIn('api:createfilew',feature_dict(item))
    def test_apistats_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'report.json'
            p.write_text(json.dumps({'behavior':{'apistats':{'1234':{'connect':3,'send':2}}}}))
            procs=parse_report(p)
            self.assertEqual(sum(procs[0]['calls'].values()),5)
    def test_summary_resolved_apis_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'report.json'
            p.write_text(json.dumps({'behavior':{'summary':{'resolved_apis':['kernel32.dll.CreateFileW','kernel32.dll.WriteFile']}}}))
            procs=parse_report(p)
            self.assertEqual(sum(procs[0]['calls'].values()),2)

    def test_stratified_split_sizes(self):
        data=[{'observed_at':'','sample_id':str(i),'label':i%2} for i in range(120)]
        groups=make_splits(data, mode='stratified', seed=42)
        self.assertEqual([len(groups[k]) for k in ('initial_train','validation','update_batch','heldout_test')],[60,18,18,24])

    def test_reject_tied_chronology(self):
        data=[{'observed_at':'2026-01-01T00:00:00+00:00','sample_id':str(i),'label':i%2} for i in range(40)]
        with self.assertRaisesRegex(ValueError,'tied'):
            make_splits(data, mode='chronological')

if __name__=='__main__': unittest.main()
