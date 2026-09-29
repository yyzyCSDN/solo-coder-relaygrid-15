from __future__ import annotations

from dataclasses import dataclass, field

from relaygrid.gis.geometry import haversine_km
from relaygrid.weather.storm_cells import contains as cell_contains

# 各来源证据在损伤置信度中的默认权重
SOURCE_WEIGHT = {
    "field_report": 0.60,      # 现场报告
    "photo": 0.55,             # 照片元数据
    "fault_indicator": 0.85,   # 故障指示器
    "asset": 0.70,             # 受影响设备
    "storm": 0.30,             # 风暴单元（佐证）
}

# 损伤点去重网格尺度：1/2000 度约 50 m
GRID_SCALE = 2000.0

ACTIVE_TASK_STATES = ("dispatched", "redirected")


class DispatchConflict(RuntimeError):
    """同一损伤点已有在岗班组时重复派遣。"""


@dataclass(frozen=True)
class FieldReport:
    report_id: str
    lat: float
    lon: float
    timestamp_s: float
    reporter: str = ""
    severity: str = "unknown"  # unknown / light / moderate / severe
    detail: str = ""
    confidence: float = 0.55


@dataclass(frozen=True)
class PhotoMetadata:
    photo_id: str
    lat: float
    lon: float
    timestamp_s: float
    report_id: str | None = None
    camera: str = ""
    damage_visible: bool = True
    confidence: float = 0.60


@dataclass(frozen=True)
class FaultSignal:
    indicator_id: str
    lat: float
    lon: float
    timestamp_s: float
    asserted: bool = True
    confidence: float = 0.80


@dataclass(frozen=True)
class AffectedAsset:
    asset_id: str
    lat: float
    lon: float
    timestamp_s: float
    kind: str = ""
    confidence: float = 0.70


@dataclass
class Evidence:
    # 原始记录（FieldReport/PhotoMetadata/...）原样保留，合并、拆分均不删除
    source_type: str
    source_id: str
    record: object
    lat: float
    lon: float
    timestamp_s: float
    weight: float


@dataclass(frozen=True)
class ConfidenceEntry:
    timestamp_s: float
    value: float
    reason: str


@dataclass
class CrewTask:
    task_id: str
    incident_id: str
    crew_id: str
    created_s: float
    lat: float
    lon: float
    status: str = "dispatched"  # dispatched / redirected / cancelled / closed
    note: str = ""


@dataclass
class DamageIncident:
    incident_id: str
    created_s: float
    evidence: list[Evidence] = field(default_factory=list)
    confidence_history: list[ConfidenceEntry] = field(default_factory=list)
    storm_cell_ids: set[str] = field(default_factory=set)
    asset_ids: set[str] = field(default_factory=set)
    task_ids: list[str] = field(default_factory=list)
    status: str = "open"  # open / merged / split / closed
    merged_into: str | None = None
    parent_id: str | None = None
    split_into: list[str] = field(default_factory=list)


def damage_point_key(lat: float, lon: float) -> str:
    return f"{round(lat * GRID_SCALE)},{round(lon * GRID_SCALE)}"


def centroid(evidence: list[Evidence]) -> tuple[float, float] | None:
    if not evidence:
        return None
    lat = sum(e.lat for e in evidence) / len(evidence)
    lon = sum(e.lon for e in evidence) / len(evidence)
    return lat, lon


def _evidence_confidence(e: Evidence) -> float:
    return max(0.0, min(1.0, getattr(e.record, "confidence", 0.6)))


def score_confidence(evidence: list[Evidence], storm_cell_ids: set[str] | None = None) -> float:
    """加权证据置信度；多源互证与风暴佐证给予有界加成。"""
    if not evidence:
        return 0.0
    total_w = sum(e.weight for e in evidence)
    avg = sum(e.weight * _evidence_confidence(e) for e in evidence) / total_w
    kinds = {e.source_type for e in evidence}
    boost = 1.0 + 0.08 * (len(kinds) - 1)
    if storm_cell_ids:
        boost += 0.05
    return min(1.0, avg * boost)


class DamageBoard:
    """损伤协同板：归并现场报告、照片、风暴单元、故障指示和受影响设备为事件。"""

    def __init__(self, cluster_km: float = 0.2, gap_s: float = 3600.0):
        self.cluster_km = cluster_km
        self.gap_s = gap_s
        self.incidents: dict[str, DamageIncident] = {}
        self.tasks: dict[str, CrewTask] = {}
        self._index: dict[tuple[str, str], str] = {}
        self._storm_cells: dict[str, object] = {}
        self._seq = 0
        self._tseq = 0

    # ------------------------------------------------------------------ ids
    def _new_incident_id(self) -> str:
        self._seq += 1
        return f"I-{self._seq}"

    def _resolve(self, incident_id: str) -> str:
        seen: set[str] = set()
        while True:
            inc = self.incidents.get(incident_id)
            if inc is None or inc.merged_into is None or incident_id in seen:
                return incident_id
            seen.add(incident_id)
            incident_id = inc.merged_into

    # ---------------------------------------------------------------- ingest
    def add_report(self, report: FieldReport) -> str:
        return self._ingest("field_report", report.report_id, report,
                            report.lat, report.lon, report.timestamp_s,
                            SOURCE_WEIGHT["field_report"])

    def add_photo(self, photo: PhotoMetadata) -> str:
        force = None
        if photo.report_id is not None:
            parent = self._index.get(("field_report", photo.report_id))
            if parent is None:
                raise ValueError(f"photo references unknown report {photo.report_id}")
            force = self._resolve(parent)
        weight = SOURCE_WEIGHT["photo"] if photo.damage_visible else 0.10
        return self._ingest("photo", photo.photo_id, photo,
                            photo.lat, photo.lon, photo.timestamp_s, weight,
                            force_incident=force)

    def add_fault_signal(self, signal: FaultSignal) -> str:
        weight = SOURCE_WEIGHT["fault_indicator"] if signal.asserted else 0.15
        return self._ingest("fault_indicator", signal.indicator_id, signal,
                            signal.lat, signal.lon, signal.timestamp_s, weight)

    def add_affected_asset(self, asset: AffectedAsset, incident_id: str | None = None) -> str:
        force = self._resolve(incident_id) if incident_id is not None else None
        return self._ingest("asset", asset.asset_id, asset,
                            asset.lat, asset.lon, asset.timestamp_s,
                            SOURCE_WEIGHT["asset"], force_incident=force)

    def link_storm_cells(self, cells: list, timestamp_s: float | None = None) -> list[str]:
        """把风暴单元关联到其覆盖范围内有证据的开放事件，返回受影响事件 id。"""
        touched: set[str] = set()
        for cell in cells:
            self._storm_cells[cell.cell_id] = cell
            for inc in self.incidents.values():
                if inc.status != "open":
                    continue
                if any(cell_contains(cell, e.lat, e.lon, e.timestamp_s) for e in inc.evidence):
                    inc.storm_cell_ids.add(cell.cell_id)
                    touched.add(inc.incident_id)
        for iid in touched:
            self._record_confidence(iid, "storm_linked", timestamp_s if timestamp_s is not None
                                    else max(e.timestamp_s for e in self.incidents[iid].evidence))
        return sorted(touched)

    def _ingest(self, source_type, source_id, record, lat, lon, timestamp_s, weight,
                force_incident: str | None = None) -> str:
        key = (source_type, source_id)
        existing = self._index.get(key)
        if existing is not None:
            return self._resolve(existing)  # 同一原始记录幂等，不重复归并
        if force_incident is not None:
            iid = force_incident
            if self.incidents[iid].status != "open":
                raise ValueError(f"incident {iid} is {self.incidents[iid].status}")
        else:
            iid = self._nearest_open(lat, lon, timestamp_s)
            if iid is None:
                iid = self._new_incident_id()
                self.incidents[iid] = DamageIncident(iid, timestamp_s)
        inc = self.incidents[iid]
        inc.evidence.append(Evidence(source_type, source_id, record, lat, lon, timestamp_s, weight))
        if source_type == "asset":
            inc.asset_ids.add(source_id)
        self._index[key] = iid
        self._record_confidence(iid, f"evidence_added:{source_type}:{source_id}", timestamp_s)
        return iid

    def _nearest_open(self, lat: float, lon: float, timestamp_s: float) -> str | None:
        best: tuple[float, str] | None = None
        for inc in self.incidents.values():
            if inc.status != "open" or not inc.evidence:
                continue
            c = centroid(inc.evidence)
            assert c is not None
            d = haversine_km(lat, lon, c[0], c[1])
            if d > self.cluster_km:
                continue
            tmin = min(e.timestamp_s for e in inc.evidence)
            tmax = max(e.timestamp_s for e in inc.evidence)
            if tmin - self.gap_s <= timestamp_s <= tmax + self.gap_s and (best is None or d < best[0]):
                best = (d, inc.incident_id)
        return best[1] if best else None

    # ------------------------------------------------------------- confidence
    def _record_confidence(self, incident_id: str, reason: str, timestamp_s: float) -> None:
        inc = self.incidents[incident_id]
        inc.confidence_history.append(
            ConfidenceEntry(timestamp_s, score_confidence(inc.evidence, inc.storm_cell_ids), reason))

    def confidence(self, incident_id: str) -> float:
        inc = self.incidents[self._resolve(incident_id)]
        return inc.confidence_history[-1].value if inc.confidence_history else 0.0

    def damage_key(self, incident_id: str) -> str | None:
        inc = self.incidents[self._resolve(incident_id)]
        if not inc.evidence:
            return None
        strongest = max(inc.evidence, key=lambda e: e.weight * _evidence_confidence(e))
        return damage_point_key(strongest.lat, strongest.lon)

    def records(self, incident_id: str) -> list[tuple[str, str, object]]:
        """取该事件自身的全部原始记录（合并后可查旧事件的留底快照，不跟随链）。"""
        inc = self.incidents[incident_id]
        return [(e.source_type, e.source_id, e.record) for e in inc.evidence]

    # --------------------------------------------------------------- dispatch
    def dispatch(self, incident_id: str, crew_id: str, timestamp_s: float,
                 task_id: str | None = None) -> str:
        iid = self._resolve(incident_id)
        inc = self.incidents[iid]
        if inc.status != "open":
            raise ValueError(f"incident {iid} is {inc.status}, cannot dispatch")
        key = self.damage_key(iid)
        for task in self.tasks.values():
            if task.status not in ACTIVE_TASK_STATES:
                continue
            if task.incident_id == iid or (
                key is not None and damage_point_key(task.lat, task.lon) == key
            ):
                raise DispatchConflict(
                    f"damage point already worked by crew {task.crew_id} on task {task.task_id}")
        self._tseq += 1
        tid = task_id or f"T-{self._tseq}"
        if tid in self.tasks:
            raise ValueError(f"duplicate task id {tid}")
        c = centroid(inc.evidence)
        assert c is not None
        task = CrewTask(tid, iid, crew_id, timestamp_s, c[0], c[1])
        self.tasks[tid] = task
        inc.task_ids.append(tid)
        return tid

    def close_task(self, task_id: str) -> None:
        self.tasks[task_id].status = "closed"

    def _active_tasks(self, incident_id: str) -> list[CrewTask]:
        return [t for t in self.tasks.values()
                if t.incident_id == incident_id and t.status in ACTIVE_TASK_STATES]

    # --------------------------------------------------------- merge / split
    def merge_incidents(self, target_id: str, other_id: str, timestamp_s: float) -> str:
        """合并两个开放事件；原始证据在双方事件下都保留快照，已有任务必须安置。"""
        a = self._resolve(target_id)
        b = self._resolve(other_id)
        if a == b:
            raise ValueError("cannot merge an incident into itself")
        inc, other = self.incidents[a], self.incidents[b]
        if inc.status != "open" or other.status != "open":
            raise ValueError("only open incidents can be merged")

        inc.evidence.extend(other.evidence)  # other.evidence 同时作为快照保留
        inc.storm_cell_ids |= other.storm_cell_ids
        inc.asset_ids |= other.asset_ids
        for e in other.evidence:
            self._index[(e.source_type, e.source_id)] = a

        # 同一损伤点只能保留一支在岗班组：最早在目标事件上的留任，其余改派或撤销
        keep_active = bool(self._active_tasks(a))
        for task in self._active_tasks(b):
            if not keep_active:
                task.status = "redirected"
                task.incident_id = a
                task.note = f"redirected by merge {b}->{a}"
                inc.task_ids.append(task.task_id)
                keep_active = True
            else:
                task.status = "cancelled"
                task.note = f"duplicate dispatch cancelled by merge {b}->{a}"

        other.status = "merged"
        other.merged_into = a
        self._record_confidence(a, f"merged:{b}", timestamp_s)
        return a

    def split_incident(self, incident_id: str,
                       groups: list[list[tuple[str, str]]], timestamp_s: float) -> list[str]:
        """按证据键分组拆分事件；原事件保留快照，在岗任务改派到最近的新事件。"""
        iid = self._resolve(incident_id)
        inc = self.incidents[iid]
        if inc.status != "open":
            raise ValueError(f"incident {iid} is {inc.status}, cannot split")
        by_key = {(e.source_type, e.source_id): e for e in inc.evidence}
        flat = [k for g in groups for k in g]
        if len(flat) != len(by_key) or set(flat) != set(by_key) or len(flat) != len(set(flat)):
            raise ValueError("split groups must partition the incident evidence exactly once")
        if len(groups) < 2:
            raise ValueError("split requires at least two groups")

        new_ids: list[str] = []
        new_centers: list[tuple[str, tuple[float, float]]] = []
        for group in groups:
            nid = self._new_incident_id()
            ev = [by_key[k] for k in group]
            child = DamageIncident(nid, timestamp_s, evidence=ev, parent_id=iid)
            child.asset_ids = {e.source_id for e in ev if e.source_type == "asset"}
            child.storm_cell_ids = self._storm_ids_for(ev)
            child.confidence_history.append(
                ConfidenceEntry(timestamp_s, score_confidence(ev, child.storm_cell_ids),
                                f"split_from:{iid}"))
            self.incidents[nid] = child
            for k in group:
                self._index[k] = nid
            new_ids.append(nid)
            new_centers.append((nid, centroid(ev)))  # type: ignore[arg-type]

        # 在岗班组跟随距离原派遣点最近的新事件，避免任务悬空或重复派工
        for task in self._active_tasks(iid):
            nid, c = min(new_centers, key=lambda nc: haversine_km(task.lat, task.lon, nc[1][0], nc[1][1]))
            task.status = "redirected"
            task.incident_id = nid
            task.lat, task.lon = c
            task.note = f"redirected by split {iid}->{nid}"
            self.incidents[nid].task_ids.append(task.task_id)

        inc.status = "split"
        inc.split_into = list(new_ids)  # evidence 原样留作快照
        return new_ids

    def _storm_ids_for(self, evidence: list[Evidence]) -> set[str]:
        ids: set[str] = set()
        for cid, cell in self._storm_cells.items():
            if any(cell_contains(cell, e.lat, e.lon, e.timestamp_s) for e in evidence):
                ids.add(cid)
        return ids
