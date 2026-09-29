from __future__ import annotations

import pytest

from relaygrid.damage import (
    DamageBoard,
    DuplicateDispatchError,
    FieldReport,
    MergeError,
    PhotoMetadata,
    SplitError,
    fuse_confidences,
    point_key,
    record_evidence,
)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_point_key_grid():
    assert point_key(40.0, -74.0) == "40.0000,-74.0000"
    assert point_key(40.00001, -74.0) == "40.0000,-74.0000"
    assert point_key(40.0001, -74.0) != "40.0000,-74.0000"


def test_fuse_confidences():
    assert fuse_confidences([]) == 0.0
    assert fuse_confidences([1.0]) == 1.0
    assert fuse_confidences([0.0, 0.0]) == 0.0
    assert fuse_confidences([0.5, 0.5]) == pytest.approx(0.75)


def test_record_evidence_weights():
    report = FieldReport("x", 0, 0, 0, "t", "a", confidence=1.0)
    assert record_evidence(report) == 1.0
    photo = PhotoMetadata("x", 0, 0, 0, "t", "cam", 1.0)
    assert record_evidence(photo) == 0.8


# ---------------------------------------------------------------------------
# ingestion / clustering
# ---------------------------------------------------------------------------

def test_ingest_groups_nearby_reports():
    board = DamageBoard(merge_radius_km=1.0, merge_window_s=3600)
    first = board.ingest_report(0, 40.0, -74.0, "downed_wire", "crew-1")
    second = board.ingest_report(60, 40.001, -74.0, "downed_wire", "crew-2")
    third = board.ingest_report(120, 41.0, -74.0, "broken_pole", "public")
    assert first.event_id == second.event_id
    assert third.event_id != first.event_id
    assert len(board.events()) == 2


def test_ingest_outside_window_creates_separate_event():
    board = DamageBoard(merge_radius_km=1.0, merge_window_s=3600)
    first = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a")
    second = board.ingest_report(7200, 40.0, -74.0, "downed_wire", "b")
    assert second.event_id != first.event_id


def test_storm_fault_equipment_evidence_raises_confidence():
    board = DamageBoard()
    event = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a", confidence=0.5)
    c0 = board.event(event.event_id).confidence
    board.ingest_storm(10, 40.001, -74.0, "CELL-7", 0.8)
    c1 = board.event(event.event_id).confidence
    board.ingest_fault(20, 40.001, -74.0, "FI-12", True)
    c2 = board.event(event.event_id).confidence
    board.ingest_equipment(30, 40.001, -74.0, "P-100", "pole", 0.8)
    c3 = board.event(event.event_id).confidence
    assert c0 < c1 < c2 < c3
    live = board.event(event.event_id)
    assert "CELL-7" in live.cell_ids
    assert "FI-12" in live.indicator_ids
    assert "P-100" in live.equipment_ids


def test_asserted_fault_beats_clear_fault():
    board = DamageBoard()
    asserted = board.ingest_fault(0, 40.0, -74.0, "FI-1", True)
    clear = board.ingest_fault(0, 41.0, -74.0, "FI-2", False)
    assert board.event(asserted.event_id).confidence > board.event(clear.event_id).confidence


def test_invalid_confidence_rejected():
    board = DamageBoard()
    with pytest.raises(ValueError):
        board.ingest_report(0, 40.0, -74.0, "x", "a", confidence=1.5)
    with pytest.raises(ValueError):
        board.ingest_photo(0, 40.0, -74.0, "x", "cam", -0.1)


# ---------------------------------------------------------------------------
# confidence traceability
# ---------------------------------------------------------------------------

def test_confidence_traceable_on_ingest():
    board = DamageBoard()
    event = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a", confidence=0.8)
    board.ingest_photo(10, 40.001, -74.0, "downed_wire", "cam", 0.9)
    live = board.event(event.event_id)
    trace = board.confidence_trace(event.event_id)
    assert len(trace) == 2
    assert trace[0].old == 0.0
    assert trace[-1].new == pytest.approx(live.confidence)
    assert all(change.reason == "ingest" for change in trace)


def test_confidence_traceable_on_merge():
    board = DamageBoard()
    first = board.ingest_report(0, 40.0, -74.0, "x", "a", confidence=0.6)
    second = board.ingest_report(7200, 40.0, -74.0, "x", "b", confidence=0.6)
    before = board.event(first.event_id).confidence
    board.merge_events([first.event_id, second.event_id], 8000)
    trace = board.confidence_trace(first.event_id)
    assert trace[-1].reason == "merge"
    assert trace[-1].old == pytest.approx(before)
    assert trace[-1].new > before


def test_confidence_traceable_on_split():
    board = DamageBoard()
    event = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a", confidence=0.8)
    board.ingest_photo(10, 40.001, -74.0, "downed_wire", "cam", 0.9)
    records = board.records_for(event.event_id)
    result = board.split_event(
        event.event_id, {records[0].record_id: "A", records[1].record_id: "B"}, 200
    )
    for new_id in result.event_ids:
        trace = board.confidence_trace(new_id)
        assert trace[0].reason == "split"
        assert trace[0].old == 0.0
        assert trace[0].new == pytest.approx(board.event(new_id).confidence)


# ---------------------------------------------------------------------------
# merges: raw records preserved, tasks handled
# ---------------------------------------------------------------------------

def test_merge_preserves_raw_records():
    board = DamageBoard()
    first = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a")
    board.ingest_photo(10, 40.001, -74.0, "downed_wire", "cam", 0.9)
    second = board.ingest_report(7200, 40.0, -74.0, "broken_pole", "b")
    total_records = len(board.raw_records())
    result = board.merge_events([first.event_id, second.event_id], 8000)
    survivor = board.event(result.event_id)
    assert len(survivor.record_ids) == 3
    assert len(board.raw_records()) == total_records
    assert board.event(second.event_id).status == "merged"
    assert board.event(second.event_id).merged_into == survivor.event_id


def test_merge_repoints_tasks():
    board = DamageBoard()
    first = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a")
    second = board.ingest_report(7200, 41.0, -74.0, "broken_pole", "b")
    third = board.ingest_report(14400, 42.0, -74.0, "broken_pole", "c")
    wo_second = board.dispatch(second.event_id, "crew-2", 200)
    wo_third = board.dispatch(third.event_id, "crew-3", 300)
    result = board.merge_events(
        [first.event_id, second.event_id, third.event_id], 15000
    )
    assert result.event_id == first.event_id
    assert sorted(result.moved_wo_ids) == sorted([wo_second.wo_id, wo_third.wo_id])
    assert result.cancelled_wo_ids == []
    assert sorted(wo.wo_id for wo in board.work_orders(first.event_id)) == sorted(
        [wo_second.wo_id, wo_third.wo_id]
    )
    assert wo_second.status == "active"
    assert wo_third.status == "active"


def test_merge_cancels_duplicate_dispatches():
    # Per-point dedup normally prevents this; the merge path is defensive.
    from relaygrid.damage.board import WorkOrder

    board = DamageBoard()
    first = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a")
    second = board.ingest_report(7200, 40.0, -74.0, "downed_wire", "b")
    wo_first = WorkOrder(
        board._next_id("WO"), first.event_id, point_key(40.0, -74.0),
        "crew-1", 100, "active",
    )
    rogue = WorkOrder(
        board._next_id("WO"), second.event_id, point_key(40.0, -74.0),
        "crew-2", 200, "active",
    )
    board._work_orders[wo_first.wo_id] = wo_first
    board._work_orders[rogue.wo_id] = rogue
    result = board.merge_events([first.event_id, second.event_id], 400)
    assert result.cancelled_wo_ids == [rogue.wo_id]
    assert rogue.status == "cancelled_duplicate"
    assert wo_first.status == "active"


def test_merge_requires_two_active_events():
    board = DamageBoard()
    first = board.ingest_report(0, 40.0, -74.0, "x", "a")
    second = board.ingest_report(10, 41.0, -74.0, "y", "b")
    with pytest.raises(MergeError):
        board.merge_events([first.event_id], 100)
    board.resolve_event(first.event_id, 50)
    with pytest.raises(MergeError):
        board.merge_events([first.event_id, second.event_id], 100)


# ---------------------------------------------------------------------------
# splits: records partitioned, tasks handled
# ---------------------------------------------------------------------------

def test_split_moves_work_order_to_owning_event():
    board = DamageBoard()
    event = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a")
    board.ingest_photo(10, 40.001, -74.0, "downed_wire", "cam", 0.9)
    records = board.records_for(event.event_id)
    wo = board.dispatch(
        event.event_id, "crew-1", 100, anchor_point=point_key(40.0, -74.0)
    )
    result = board.split_event(
        event.event_id, {records[0].record_id: "A", records[1].record_id: "B"}, 200
    )
    assert len(result.event_ids) == 2
    assert result.moved_wo_ids == [wo.wo_id]
    assert result.orphaned_wo_ids == []
    assert board.event(wo.event_id).point_keys == {point_key(40.0, -74.0)}
    assert board.event(event.event_id).status == "split"


def test_split_orphans_work_order_without_host():
    # A WO whose point belongs to no split group is orphaned (defensive path).
    from relaygrid.damage.board import WorkOrder

    board = DamageBoard()
    event = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a")
    board.ingest_photo(10, 40.001, -74.0, "downed_wire", "cam", 0.9)
    records = board.records_for(event.event_id)
    wo = WorkOrder(
        board._next_id("WO"), event.event_id, "99.0000,-74.0000",
        "crew-1", 100, "active",
    )
    board._work_orders[wo.wo_id] = wo
    result = board.split_event(
        event.event_id, {records[0].record_id: "A", records[1].record_id: "B"}, 200
    )
    assert result.orphaned_wo_ids == [wo.wo_id]
    assert wo.status == "orphaned"
    assert board.work_orders(event.event_id) == []


def test_split_requires_full_partition():
    board = DamageBoard()
    event = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a")
    board.ingest_photo(10, 40.001, -74.0, "downed_wire", "cam", 0.9)
    records = board.records_for(event.event_id)
    with pytest.raises(SplitError):
        board.split_event(event.event_id, {records[0].record_id: "A"}, 200)
    with pytest.raises(SplitError):
        board.split_event(
            event.event_id, {records[0].record_id: "A", records[1].record_id: "A"}, 200
        )


def test_split_accepts_equipment_id_assignment():
    board = DamageBoard()
    event = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a")
    board.ingest_equipment(10, 40.001, -74.0, "P-100", "pole", 0.8)
    result = board.split_event(
        event.event_id, {event.record_ids[0]: "A", "P-100": "B"}, 200
    )
    assert len(result.event_ids) == 2
    assert {len(board.records_for(eid)) for eid in result.event_ids} == {1}


# ---------------------------------------------------------------------------
# dispatch dedup / lifecycle
# ---------------------------------------------------------------------------

def test_dispatch_dedup_per_damage_point():
    board = DamageBoard()
    first = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a")
    second = board.ingest_report(10, 41.0, -74.0, "broken_pole", "b")
    board.dispatch(first.event_id, "crew-1", 100)
    with pytest.raises(DuplicateDispatchError):
        board.dispatch(first.event_id, "crew-2", 110)
    board.dispatch(second.event_id, "crew-2", 120)
    # Same damage point, separate event (outside window) -> still blocked.
    third = board.ingest_report(7200, 40.0, -74.0, "downed_wire", "c")
    with pytest.raises(DuplicateDispatchError):
        board.dispatch(third.event_id, "crew-3", 210)
    assert board.dispatched_points() == {point_key(40.0, -74.0), point_key(41.0, -74.0)}


def test_resolve_and_complete():
    board = DamageBoard()
    event = board.ingest_report(0, 40.0, -74.0, "downed_wire", "a")
    wo = board.dispatch(event.event_id, "crew-1", 100)
    assert board.event(event.event_id).status == "dispatched"
    board.complete_work_order(wo.wo_id, 200)
    assert wo.status == "completed"
    board.resolve_event(event.event_id, 300)
    assert board.event(event.event_id).status == "resolved"
    assert board.events() == []
    assert len(board.events(include_closed=True)) == 1
