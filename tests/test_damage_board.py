import pytest

from relaygrid.damage import (
    AffectedAsset,
    DamageBoard,
    DispatchConflict,
    FieldReport,
    FaultSignal,
    PhotoMetadata,
    damage_point_key,
    score_confidence,
)
from relaygrid.weather.storm_cells import StormCell


def _board():
    return DamageBoard(cluster_km=0.2, gap_s=3600)


def test_nearby_reports_merge_far_separate_and_records_retained():
    b = _board()
    i1 = b.add_report(FieldReport("r1", 30.0, 120.0, 0, "lineman"))
    i2 = b.add_report(FieldReport("r2", 30.0003, 120.0003, 100, "patrol"))
    i3 = b.add_report(FieldReport("r3", 31.0, 121.0, 100, "patrol"))
    assert i1 == i2
    assert i3 != i1
    sources = {sid for _, sid, _ in b.records(i1)}
    assert sources == {"r1", "r2"}  # 原始记录均保留


def test_photo_and_asset_and_fault_signal_join_incident():
    b = _board()
    iid = b.add_report(FieldReport("r1", 30.0, 120.0, 0))
    assert b.add_photo(PhotoMetadata("p1", 30.0002, 120.0002, 60, report_id="r1")) == iid
    assert b.add_fault_signal(FaultSignal("fi-7", 30.0004, 120.0004, 30)) == iid
    assert b.add_affected_asset(AffectedAsset("pole-9", 30.0001, 120.0001, 20)) == iid
    assert b.incidents[iid].asset_ids == {"pole-9"}
    assert {typ for typ, _, _ in b.records(iid)} == {"field_report", "photo", "fault_indicator", "asset"}


def test_photo_unknown_report_rejected():
    b = _board()
    with pytest.raises(ValueError):
        b.add_photo(PhotoMetadata("p1", 30.0, 120.0, 0, report_id="ghost"))


def test_confidence_history_traceable_and_rises_with_corroboration():
    b = _board()
    iid = b.add_report(FieldReport("r1", 30.0, 120.0, 0))
    first = b.confidence(iid)
    b.add_fault_signal(FaultSignal("fi-7", 30.0002, 120.0002, 30))
    second = b.confidence(iid)
    b.link_storm_cells([StormCell("s1", 30.0, 120.0, 5, 25, 40, 0, 600)])
    third = b.confidence(iid)
    assert 0.0 < first < second < third <= 1.0
    history = b.incidents[iid].confidence_history
    assert [h.value for h in history] == [first, second, third]
    assert history[-1].reason == "storm_linked"


def test_ingestion_is_idempotent():
    b = _board()
    i1 = b.add_report(FieldReport("r1", 30.0, 120.0, 0))
    i2 = b.add_report(FieldReport("r1", 30.0, 120.0, 0))
    assert i1 == i2
    assert len(b.incidents[i1].evidence) == 1


def test_same_damage_point_cannot_be_dispatched_twice():
    b = _board()
    iid = b.add_report(FieldReport("r1", 30.0, 120.0, 0))
    t1 = b.dispatch(iid, "crew-a", 100)
    with pytest.raises(DispatchConflict):
        b.dispatch(iid, "crew-b", 200)
    # 另一个独立事件，但其最强证据落在同一损伤网格，同样拦截全局重复派遣
    other = b.add_report(FieldReport("r2", 31.0, 121.0, 0, confidence=0.3))
    b.add_affected_asset(AffectedAsset("pole-9", 30.00001, 120.00001, 20, confidence=0.9),
                         incident_id=other)
    assert b.damage_key(other) == b.damage_key(iid)
    with pytest.raises(DispatchConflict):
        b.dispatch(other, "crew-c", 200)
    b.close_task(t1)
    assert b.dispatch(iid, "crew-b", 300)  # 完工后可再派


def test_merge_keeps_records_and_handles_existing_tasks():
    b = DamageBoard(cluster_km=5.0)
    a = b.add_report(FieldReport("ra", 30.0, 120.0, 0))
    cid = b.add_report(FieldReport("rc", 30.2, 120.2, 10))
    b.dispatch(a, "crew-a", 100)
    b.dispatch(cid, "crew-c", 200)
    merged = b.merge_incidents(a, cid, 300)
    assert merged == a
    assert {sid for _, sid, _ in b.records(a)} == {"ra", "rc"}
    # 被合并事件的原始记录快照仍在
    assert {sid for _, sid, _ in b.records(cid)} == {"rc"}
    assert b.incidents[cid].status == "merged" and b.incidents[cid].merged_into == a
    tasks = sorted(b.tasks.values(), key=lambda t: t.task_id)
    statuses = {t.task_id: t.status for t in tasks}
    assert "dispatched" in statuses.values() and "cancelled" in statuses.values()
    # 旧 id 自动解析到存活事件
    assert b._resolve(cid) == a


def test_merge_without_task_conflict_redirects():
    b = DamageBoard(cluster_km=5.0)
    a = b.add_report(FieldReport("ra", 30.0, 120.0, 0))
    cid = b.add_report(FieldReport("rc", 30.2, 120.2, 10))
    tid = b.dispatch(cid, "crew-c", 200)
    b.merge_incidents(a, cid, 300)
    task = b.tasks[tid]
    assert task.status == "redirected" and task.incident_id == a


def test_split_partitions_evidence_and_moves_task():
    b = DamageBoard(cluster_km=50.0)
    iid = b.add_report(FieldReport("r1", 30.0, 120.0, 0))
    b.add_report(FieldReport("r2", 30.3, 120.3, 10))
    b.add_fault_signal(FaultSignal("fi", 30.0001, 120.0001, 5))
    tid = b.dispatch(iid, "crew-a", 100)
    keys = [(e.source_type, e.source_id) for e in b.incidents[iid].evidence]
    g1, g2 = [keys[0], keys[2]], [keys[1]]
    n1, n2 = b.split_incident(iid, [g1, g2], 400)
    assert b.incidents[iid].status == "split"
    assert b.incidents[iid].split_into == [n1, n2]
    assert len(b.records(n1)) == 2 and len(b.records(n2)) == 1
    # 原事件快照保留全部证据
    assert len(b.records(iid)) == 3
    task = b.tasks[tid]
    assert task.status == "redirected" and task.incident_id in (n1, n2)
    # 子事件可独立派工，但同一损伤点仍受全局保护
    child_with_task = task.incident_id
    with pytest.raises(DispatchConflict):
        b.dispatch(child_with_task, "crew-b", 500)


def test_split_must_partition_exactly():
    b = _board()
    iid = b.add_report(FieldReport("r1", 30.0, 120.0, 0))
    b.add_report(FieldReport("r2", 30.0003, 120.0003, 10))
    keys = [(e.source_type, e.source_id) for e in b.incidents[iid].evidence]
    with pytest.raises(ValueError):
        b.split_incident(iid, [[keys[0]]], 400)  # 漏证据
    with pytest.raises(ValueError):
        b.split_incident(iid, [keys, keys[:1]], 400)  # 重复


def test_damage_key_grid():
    assert damage_point_key(30.0, 120.0) == damage_point_key(30.00001, 120.00001)
    assert damage_point_key(30.0, 120.0) != damage_point_key(30.001, 120.001)


def test_score_confidence_basic():
    assert score_confidence([]) == 0.0
    b = _board()
    iid = b.add_report(FieldReport("r1", 30.0, 120.0, 0, confidence=1.0))
    assert 0.0 < b.confidence(iid) <= 1.0
