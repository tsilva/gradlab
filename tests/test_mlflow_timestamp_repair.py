from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sqlite3
import json
import hashlib

import pytest

spec = spec_from_file_location('repair', Path(__file__).resolve().parents[1]/'ops/mlflow/repair_timestamps.py')
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_repair_is_exact_terminal_scoped_and_idempotent(tmp_path):
    db = tmp_path/'tracking.db'
    with sqlite3.connect(db) as conn:
        conn.executescript('''CREATE TABLE runs(run_uuid TEXT,status TEXT);
        CREATE TABLE tags(run_uuid TEXT,key TEXT,value TEXT);
        CREATE TABLE metrics(key TEXT,value REAL,timestamp INTEGER,run_uuid TEXT,step INTEGER,is_nan BOOLEAN);
        CREATE TABLE latest_metrics(key TEXT,value REAL,timestamp INTEGER,step INTEGER,is_nan BOOLEAN,run_uuid TEXT);
        INSERT INTO runs VALUES('target','FINISHED'),('other','RUNNING');
        INSERT INTO tags VALUES('target','gradlab.run_id','gradlab-test');
        INSERT INTO metrics VALUES('ops/sequence',1,1001,'target',1,0),('ops/outbox/count',0,1001,'target',128,0),('other',2,1001,'other',128,0);''')
    snapshot = tmp_path/'snapshot.db'
    snapshot.write_bytes(db.read_bytes())
    snapshot.with_suffix('.db.json').write_text(json.dumps({'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest()}))
    evidence = {'manifest':{'created_at':'1970-01-01T00:00:01Z','run_id':'gradlab-test'},'service_run_id':'target',
                'events':[{'event_seq':1,'kind':'history','source':'orchestration:health','step':128,'created_at':10.5,'payload':{'ops/outbox/count':0}}]}
    assert module.repair(db,evidence,snapshot)['changed'] == 2
    assert module.repair(db,evidence,snapshot)['changed'] == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT timestamp,step FROM metrics WHERE run_uuid='target'").fetchall() == [(10500,1),(10500,1)]
        assert conn.execute("SELECT timestamp,step FROM metrics WHERE run_uuid='other'").fetchall() == [(1001,128)]
    evidence['events'][0]['payload']['ops/outbox/count'] = 9
    with pytest.raises(ValueError,match='differs'):
        module.repair(db,evidence,snapshot)
