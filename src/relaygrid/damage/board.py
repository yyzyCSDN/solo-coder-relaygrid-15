"""损伤协同板 — damage coordination board.

Field reports (现场报告), photo metadata (照片元数据), storm-cell context
(风暴单元), fault indicators (故障指示), and affected equipment (受影响设备)
are ingested as immutable raw records and grouped into damage events.

Guarantees:

* 原始记录不丢 — raw records are never discarded; similar events merge and
  large events split by re-grouping references to the original records.
* 置信度变化可追溯 — every confidence change is appended to an audit trail
  (ingest / merge / split) with timestamp, old value, new value, and trigger.
* 同一损伤点不能重复派遣班组 — crew dispatch is deduplicated per damage
  point (a grid-keyed point location).
* 拆分或合并事件要处理已有任务 — merges re-point work orders to the survivor
  and cancel duplicate dispatches; splits move work orders to the new event
  that owns their point, and orphan the rest with a visible status.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from relaygrid.gis.geometry import haversine_km


# ---------------------------------------------------------------------------
# Raw evidence records (frozen, immutable, never deleted)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldReport:
    """A field report (现场报告) about suspected damage."""

    record_id: str
    ts: float
    lat: float
    lon: float
    damage_type: str
    reporter: str
    severity: float = 0.5
    confidence: float = 0.85
    note: str = ""


@dataclass(frozen=True)
class PhotoMetadata:
    """Photo metadata (照片元数据) with a classifier confidence."""

    record_id: str
    ts: float
    lat: float
    lon: float
    damage_type: str
    classifier: str
    confidence: float
    report_id: str | None = None


@dataclass(frozen=True)
class StormCellObservation:
    """Storm-cell (风暴单元) context: damage happened inside a storm footprint."""

    record_id: str
    ts: float
    lat: float
    lon: float
    cell_id: str
    severity: float
    distance_km: float = 0.0


@dataclass(frozen=True)
class FaultIndicatorObservation:
    """Fault-indicator (故障指示) observation."""

    record_id: str
    ts: float
    lat: float
    lon: float
    indicator_id: str
    asserted: bool
    confidence: float | None = None


@dataclass(frozen=True)
class AffectedEquipment:
    """Affected equipment (受影响设备) at a damage point."""

    record_id: str
    ts: float
    lat: float
    lon: float
    equipment_id: str
    equipment_type: str
    damage_severity: float = 0.5
    confidence: float = 0.8


# ---------------------------------------------------------------------------
# Derived state
# ---------------------------------------------------------------------------


@dataclass
class DamageEvent:
    """A damage event grouping raw evidence around one or more points."""

    event_id: str
    created_ts: float
    updated_ts: float
    lat: float
    lon: float
    damage_type: str
    status: str = "open"  # open | dispatched | in_progress | resolved | merged | split
    record_ids: list[str] = field(default_factory=list)
    equipment_ids: list[str] = field(default_factory=list)
    cell_ids: list[str] = field(default_factory=list)
    indicator_ids: list[str] = field(default_factory=list)
    point_keys: set[str] = field(default_factory=set)
    confidence: float = 0.0
    merged_into: str | None = None
    split_into: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConfidenceChange:
    """One entry in an event's confidence audit trail (置信度变化可追溯)."""

    ts: float
    old: float
    new: float
    reason: str  # "ingest" | "merge" | "split"
    trigger: str  # record id / event id that caused the change


@dataclass
class WorkOrder:
    """A crew work order (任务) tied to a damage point."""

    wo_id: str
    event_id: str
    point_key: str
    crew_id: str
    ts: float
    status: str = "active"  # active | completed | cancelled_duplicate | orphaned
    note: str = ""


@dataclass(frozen=True)
class MergeResult:
    event_id: str
    moved_wo_ids: list[str]
    cancelled_wo_ids: list[str]


@dataclass(frozen=True)
class SplitResult:
    event_ids: list[str]
    moved_wo_ids: list[str]
    orphaned_wo_ids: list[str]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DuplicateDispatchError(RuntimeError):
    """Raised when a damage point already has an active crew dispatch."""


class MergeError(RuntimeError):
    """Raised when events cannot be merged."""


class SplitError(RuntimeError):
    """Raised when an event cannot be split with the given partition."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

#: Per-source trust weights used when fusing event confidence.
_SOURCE_WEIGHTS: dict[type, float] = {
    FieldReport: 1.0,
    PhotoMetadata: 0.8,
    StormCellObservation: 0.3,
    FaultIndicatorObservation: 0.9,
    AffectedEquipment: 0.9,
}


def point_key(lat: float, lon: float, decimals: int = 4) -> str:
    """Canonical damage-point key; a 4-decimal grid is ~11 m across."""
    return f"{round(lat, decimals):.{decimals}f},{round(lon, decimals):.{decimals}f}"


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def record_evidence(record) -> float:
    """Weighted evidence contribution of a single raw record."""
    weight = _SOURCE_WEIGHTS[type(record)]
    if isinstance(record, StormCellObservation):
        confidence = record.severity
    elif isinstance(record, FaultIndicatorObservation):
        confidence = (
            record.confidence
            if record.confidence is not None
            else (0.7 if record.asserted else 0.2)
        )
    else:
        confidence = record.confidence
    return weight * _clamp01(confidence)


def fuse_confidences(confidences: Iterable[float]) -> float:
    """Noisy-or fusion: multiple corroborating sources raise confidence."""
    p_none = 1.0
    for confidence in confidences:
        p_none *= 1.0 - _clamp01(confidence)
    return 1.0 - p_none


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------


class DamageBoard:
    """In-memory damage coordination board.

    Ingested records auto-attach to a nearby open event (spatial-temporal
    clustering); explicit :meth:`merge_events` / :meth:`split_event` regroup
    evidence while preserving every raw record and re-pointing work orders.
    """

    _ACTIVE_STATUSES = ("open", "dispatched", "in_progress")

    def __init__(
        self,
        merge_radius_km: float = 0.5,
        merge_window_s: float = 3600.0,
        point_decimals: int = 4,
    ):
        if merge_radius_km < 0 or merge_window_s < 0:
            raise ValueError("merge radius/window must be non-negative")
        self.merge_radius_km = merge_radius_km
        self.merge_window_s = merge_window_s
        self.point_decimals = point_decimals
        self._records: dict[str, object] = {}
        self._events: dict[str, DamageEvent] = {}
        self._work_orders: dict[str, WorkOrder] = {}
        self._traces: dict[str, list[ConfidenceChange]] = {}
        self._seq = 0

    # ----- identifiers -----------------------------------------------------

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq:04d}"

    @staticmethod
    def _validate_unit(value: float, name: str) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    # ----- ingestion -------------------------------------------------------

    def ingest_report(
        self,
        ts: float,
        lat: float,
        lon: float,
        damage_type: str,
        reporter: str,
        severity: float = 0.5,
        confidence: float = 0.85,
        note: str = "",
    ) -> DamageEvent:
        self._validate_unit(severity, "severity")
        self._validate_unit(confidence, "confidence")
        record = FieldReport(
            self._next_id("R"), ts, lat, lon, damage_type, reporter,
            _clamp01(severity), confidence, note,
        )
        return self._attach(record)

    def ingest_photo(
        self,
        ts: float,
        lat: float,
        lon: float,
        damage_type: str,
        classifier: str,
        confidence: float,
        report_id: str | None = None,
    ) -> DamageEvent:
        self._validate_unit(confidence, "confidence")
        record = PhotoMetadata(
            self._next_id("R"), ts, lat, lon, damage_type, classifier,
            confidence, report_id,
        )
        return self._attach(record)

    def ingest_storm(
        self,
        ts: float,
        lat: float,
        lon: float,
        cell_id: str,
        severity: float,
        distance_km: float = 0.0,
    ) -> DamageEvent:
        self._validate_unit(severity, "severity")
        record = StormCellObservation(
            self._next_id("R"), ts, lat, lon, cell_id, severity,
            max(0.0, distance_km),
        )
        return self._attach(record)

    def ingest_fault(
        self,
        ts: float,
        lat: float,
        lon: float,
        indicator_id: str,
        asserted: bool,
        confidence: float | None = None,
    ) -> DamageEvent:
        if confidence is not None:
            self._validate_unit(confidence, "confidence")
        record = FaultIndicatorObservation(
            self._next_id("R"), ts, lat, lon, indicator_id, asserted, confidence
        )
        return self._attach(record)

    def ingest_equipment(
        self,
        ts: float,
        lat: float,
        lon: float,
        equipment_id: str,
        equipment_type: str,
        damage_severity: float = 0.5,
        confidence: float = 0.8,
    ) -> DamageEvent:
        self._validate_unit(damage_severity, "damage_severity")
        self._validate_unit(confidence, "confidence")
        record = AffectedEquipment(
            self._next_id("R"), ts, lat, lon, equipment_id, equipment_type,
            _clamp01(damage_severity), confidence,
        )
        return self._attach(record)

    def _attach(self, record) -> DamageEvent:
        if record.record_id in self._records:
            raise ValueError(f"duplicate raw record {record.record_id}")
        self._records[record.record_id] = record
        host = self._find_host(record)
        if host is None:
            host = self._new_event(record)
        else:
            self._add_record(host, record, "ingest")
        return host

    def _find_host(self, record) -> DamageEvent | None:
        """Nearest open event within the spatial-temporal clustering window."""
        best: DamageEvent | None = None
        best_distance = float("inf")
        for event in self._events.values():
            if event.status not in self._ACTIVE_STATUSES:
                continue
            time_gap = min(
                abs(record.ts - self._records[rid].ts) for rid in event.record_ids
            )
            if time_gap > self.merge_window_s:
                continue
            distance = haversine_km(record.lat, record.lon, event.lat, event.lon)
            if distance <= self.merge_radius_km and distance < best_distance:
                best = event
                best_distance = distance
        return best

    def _new_event(self, record) -> DamageEvent:
        event = DamageEvent(
            self._next_id("E"), record.ts, record.ts,
            record.lat, record.lon, self._damage_type(record),
        )
        event.point_keys.add(point_key(record.lat, record.lon, self.point_decimals))
        self._events[event.event_id] = event
        self._traces[event.event_id] = []
        self._add_record(event, record, "ingest")
        return event

    def _add_record(self, event: DamageEvent, record, reason: str) -> None:
        event.record_ids.append(record.record_id)
        event.point_keys.add(point_key(record.lat, record.lon, self.point_decimals))
        if isinstance(record, AffectedEquipment):
            event.equipment_ids.append(record.equipment_id)
        elif isinstance(record, StormCellObservation):
            event.cell_ids.append(record.cell_id)
        elif isinstance(record, FaultIndicatorObservation):
            event.indicator_ids.append(record.indicator_id)
        event.updated_ts = max(event.updated_ts, record.ts)
        event.lat, event.lon = self._centroid(event)
        event.damage_type = self._majority_damage_type(event)
        old_confidence = event.confidence
        event.confidence = self._compute_confidence(event)
        self._traces[event.event_id].append(
            ConfidenceChange(record.ts, old_confidence, event.confidence, reason, record.record_id)
        )

    # ----- confidence ------------------------------------------------------

    def _compute_confidence(self, event: DamageEvent) -> float:
        return fuse_confidences(
            record_evidence(self._records[rid]) for rid in event.record_ids
        )

    def _centroid(self, event: DamageEvent) -> tuple[float, float]:
        records = [self._records[rid] for rid in event.record_ids]
        lat = sum(r.lat for r in records) / len(records)
        lon = sum(r.lon for r in records) / len(records)
        return (lat, lon)

    @staticmethod
    def _damage_type(record) -> str:
        if isinstance(record, (FieldReport, PhotoMetadata)):
            return record.damage_type
        return "unknown"

    def _majority_damage_type(self, event: DamageEvent) -> str:
        votes = Counter(
            self._records[rid].damage_type
            for rid in event.record_ids
            if isinstance(self._records[rid], (FieldReport, PhotoMetadata))
        )
        return votes.most_common(1)[0][0] if votes else "unknown"

    # ----- merges ----------------------------------------------------------

    def merge_events(
        self, event_ids: list[str], ts: float, reason: str = "merge"
    ) -> MergeResult:
        """Merge events into the earliest-created survivor.

        Raw records are unioned (原始记录不丢), confidence is recomputed with a
        trace entry, and existing work orders are re-pointed: duplicate active
        dispatches for the same damage point are cancelled (earliest kept).
        """
        ids = list(dict.fromkeys(event_ids))
        if len(ids) < 2:
            raise MergeError("need at least two distinct events to merge")
        events: list[DamageEvent] = []
        for event_id in ids:
            event = self._events.get(event_id)
            if event is None:
                raise MergeError(f"unknown event {event_id}")
            if event.status not in self._ACTIVE_STATUSES:
                raise MergeError(f"event {event_id} is {event.status}, cannot merge")
            events.append(event)

        survivor = min(events, key=lambda e: e.created_ts)
        moved, cancelled = self._repoint_on_merge(survivor, events)

        for event in events:
            if event is survivor:
                continue
            survivor.record_ids.extend(event.record_ids)
            survivor.equipment_ids.extend(event.equipment_ids)
            survivor.cell_ids.extend(event.cell_ids)
            survivor.indicator_ids.extend(event.indicator_ids)
            survivor.point_keys |= event.point_keys
            event.status = "merged"
            event.merged_into = survivor.event_id

        survivor.updated_ts = max(survivor.updated_ts, ts)
        survivor.lat, survivor.lon = self._centroid(survivor)
        survivor.damage_type = self._majority_damage_type(survivor)
        old_confidence = survivor.confidence
        survivor.confidence = self._compute_confidence(survivor)
        self._traces[survivor.event_id].append(
            ConfidenceChange(
                ts, old_confidence, survivor.confidence, reason,
                ",".join(e.event_id for e in events),
            )
        )
        return MergeResult(survivor.event_id, moved, cancelled)

    def _repoint_on_merge(
        self, survivor: DamageEvent, events: list[DamageEvent]
    ) -> tuple[list[str], list[str]]:
        """Move work orders to the survivor; cancel duplicate point dispatches."""
        active_by_point: dict[str, list[WorkOrder]] = defaultdict(list)
        for event in events:
            for wo in self._work_orders.values():
                if wo.event_id == event.event_id and wo.status == "active":
                    active_by_point[wo.point_key].append(wo)

        cancelled: list[str] = []
        for point_key, orders in active_by_point.items():
            orders.sort(key=lambda w: w.ts)
            for duplicate in orders[1:]:
                duplicate.status = "cancelled_duplicate"
                duplicate.note = f"merged into {survivor.event_id}"
                # Re-point so all orders for the merged entity stay traceable.
                duplicate.event_id = survivor.event_id
                cancelled.append(duplicate.wo_id)

        moved: list[str] = []
        for event in events:
            if event is survivor:
                continue
            for wo in self._work_orders.values():
                if wo.event_id == event.event_id and wo.status == "active":
                    wo.event_id = survivor.event_id
                    moved.append(wo.wo_id)
        return moved, cancelled

    # ----- splits ----------------------------------------------------------

    def split_event(
        self,
        event_id: str,
        assignments: dict[str, str],
        ts: float,
        reason: str = "split",
    ) -> SplitResult:
        """Split an event into new events keyed by group id.

        ``assignments`` maps every raw record id (or equipment id) of the event
        to a new group. Records are never deleted (原始记录不丢); work orders
        move to the new event owning their damage point, and orders whose point
        lands in no event are orphaned (已有任务必须处理).
        """
        event = self._events.get(event_id)
        if event is None:
            raise SplitError(f"unknown event {event_id}")
        if event.status not in self._ACTIVE_STATUSES:
            raise SplitError(f"event {event_id} is {event.status}, cannot split")

        record_groups: dict[str, list[str]] = defaultdict(list)
        for key, group in assignments.items():
            record_id = self._resolve_record_key(event, key)
            record_groups[group].append(record_id)

        assigned = {rid for rids in record_groups.values() for rid in rids}
        if assigned != set(event.record_ids):
            missing = sorted(set(event.record_ids) - assigned)
            extra = sorted(assigned - set(event.record_ids))
            raise SplitError(
                f"split must assign every record exactly once; missing={missing}, extra={extra}"
            )
        if len(record_groups) < 2:
            raise SplitError("split requires at least two non-empty groups")

        new_events = [
            self._spawn_from_records(event, rids, ts, reason)
            for rids in record_groups.values()
        ]
        moved, orphaned = self._repoint_on_split(event, new_events)

        event.status = "split"
        event.split_into = [e.event_id for e in new_events]
        return SplitResult(
            [e.event_id for e in new_events], moved, orphaned
        )

    def _resolve_record_key(self, event: DamageEvent, key: str) -> str:
        if key in event.record_ids:
            return key
        matches = [
            rid
            for rid in event.record_ids
            if isinstance(self._records[rid], AffectedEquipment)
            and self._records[rid].equipment_id == key
        ]
        if len(matches) == 1:
            return matches[0]
        raise SplitError(f"id {key!r} is not a record or equipment of event {event.event_id}")

    def _spawn_from_records(
        self, parent: DamageEvent, record_ids: list[str], ts: float, reason: str
    ) -> DamageEvent:
        first = self._records[record_ids[0]]
        child = DamageEvent(
            self._next_id("E"), ts, ts, first.lat, first.lon,
            self._damage_type(first),
        )
        self._events[child.event_id] = child
        self._traces[child.event_id] = []
        for record_id in record_ids:
            self._add_record(child, self._records[record_id], reason)
        return child

    def _repoint_on_split(
        self, parent: DamageEvent, new_events: list[DamageEvent]
    ) -> tuple[list[str], list[str]]:
        moved: list[str] = []
        orphaned: list[str] = []
        for wo in self._work_orders.values():
            if wo.event_id != parent.event_id or wo.status != "active":
                continue
            hosts = [e for e in new_events if wo.point_key in e.point_keys]
            if len(hosts) > 1:
                raise SplitError(
                    f"work order {wo.wo_id} point {wo.point_key} is claimed by multiple split events"
                )
            if len(hosts) == 1:
                wo.event_id = hosts[0].event_id
                moved.append(wo.wo_id)
            else:
                wo.status = "orphaned"
                wo.note = f"split from {parent.event_id}"
                orphaned.append(wo.wo_id)
        return moved, orphaned

    # ----- dispatch --------------------------------------------------------

    def dispatch(
        self,
        event_id: str,
        crew_id: str,
        ts: float,
        note: str = "",
        anchor_point: str | None = None,
    ) -> WorkOrder:
        """Dispatch a crew to an event.

        Raises :class:`DuplicateDispatchError` if the anchor damage point
        already has an active dispatch (同一损伤点不能重复派遣班组).

        ``anchor_point`` anchors the order to one damage point of the event; by
        default the event's primary point (the point nearest the centroid) is
        used. A second crew may be dispatched to a *different* damage point of
        the same event.
        """
        event = self._events.get(event_id)
        if event is None:
            raise DuplicateDispatchError(f"unknown event {event_id}")
        if event.status not in self._ACTIVE_STATUSES:
            raise DuplicateDispatchError(f"event {event_id} is {event.status}")
        if anchor_point is None:
            centroid_key = point_key(event.lat, event.lon, self.point_decimals)
            anchor_point = (
                centroid_key
                if centroid_key in event.point_keys
                else sorted(event.point_keys)[0]
            )
        elif anchor_point not in event.point_keys:
            raise ValueError(f"point {anchor_point} is not part of event {event_id}")

        active_points = {
            wo.point_key for wo in self._work_orders.values() if wo.status == "active"
        }
        if anchor_point in active_points:
            raise DuplicateDispatchError(
                f"active dispatch already exists for damage point {anchor_point}"
            )

        wo = WorkOrder(
            self._next_id("WO"), event_id, anchor_point,
            crew_id, ts, "active", note,
        )
        self._work_orders[wo.wo_id] = wo
        if event.status == "open":
            event.status = "dispatched"
        return wo

    def complete_work_order(self, wo_id: str, ts: float, note: str = "") -> WorkOrder:
        wo = self._work_orders.get(wo_id)
        if wo is None:
            raise KeyError(wo_id)
        if wo.status != "active":
            raise ValueError(f"work order {wo_id} is {wo.status}, cannot complete")
        wo.status = "completed"
        wo.note = note
        return wo

    def resolve_event(self, event_id: str, ts: float, note: str = "") -> DamageEvent:
        event = self._events.get(event_id)
        if event is None:
            raise KeyError(event_id)
        if event.status in ("merged", "split"):
            raise MergeError(f"event {event_id} is {event.status}")
        event.status = "resolved"
        event.updated_ts = max(event.updated_ts, ts)
        return event

    # ----- accessors -------------------------------------------------------

    def events(self, include_closed: bool = False) -> list[DamageEvent]:
        if include_closed:
            return list(self._events.values())
        return [
            e for e in self._events.values()
            if e.status not in ("merged", "split", "resolved", "closed")
        ]

    def event(self, event_id: str) -> DamageEvent:
        event = self._events.get(event_id)
        if event is None:
            raise KeyError(event_id)
        return event

    def raw_records(self) -> list:
        return list(self._records.values())

    def records_for(self, event_id: str) -> list:
        event = self.event(event_id)
        return [self._records[rid] for rid in event.record_ids]

    def work_orders(self, event_id: str | None = None) -> list[WorkOrder]:
        orders = list(self._work_orders.values())
        if event_id is not None:
            orders = [wo for wo in orders if wo.event_id == event_id]
        return orders

    def dispatched_points(self) -> set[str]:
        """Damage points that currently have an active crew dispatch."""
        return {wo.point_key for wo in self._work_orders.values() if wo.status == "active"}

    def confidence_trace(self, event_id: str) -> list[ConfidenceChange]:
        self.event(event_id)
        return list(self._traces[event_id])
