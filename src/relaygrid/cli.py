from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from relaygrid.damage import DamageBoard, DuplicateDispatchError
from relaygrid.models import Branch, Bus, FeederSnapshot, SwitchState
from relaygrid.sim.engine import GridSimulator


def demo_summary() -> dict[str, object]:
    """Run a deterministic switching example and return its public summary."""
    buses = {bus_id: Bus(bus_id, 12.47) for bus_id in "ABCD"}
    branches = {
        "ab": Branch("ab", "A", "B", 0.1, 0.1, 300.0),
        "bc": Branch("bc", "B", "C", 0.1, 0.1, 300.0),
        "cd": Branch("cd", "C", "D", 0.1, 0.1, 300.0, True, False),
        "ad": Branch("ad", "A", "D", 0.1, 0.1, 300.0, True, False),
    }
    snapshot = FeederSnapshot(
        buses,
        branches,
        {"cd": SwitchState.OPEN, "ad": SwitchState.OPEN},
    )
    simulator = GridSimulator(snapshot)
    energized_before = simulator.energized_bus_count()
    simulator.operate_switch("cd", SwitchState.CLOSED)
    return {
        "energized_after": simulator.energized_bus_count(),
        "energized_before": energized_before,
        "events": len(simulator.log.events),
        "project": "RelayGrid",
        "switch": "cd",
        "switch_state": snapshot.switch_state["cd"].value,
    }


def damage_demo_summary() -> dict[str, object]:
    """Run a deterministic damage coordination board example."""
    board = DamageBoard(merge_radius_km=0.5, merge_window_s=3600.0)

    # Five corroborating sources around the same damage point -> one event.
    first = board.ingest_report(0.0, 40.7128, -74.0060, "downed_wire", "crew-1", severity=0.9)
    board.ingest_photo(120.0, 40.7135, -74.0050, "downed_wire", "vision-ml", 0.92)
    board.ingest_storm(300.0, 40.7140, -74.0040, "CELL-7", 0.6)
    board.ingest_fault(360.0, 40.7130, -74.0065, "FI-12", True)
    board.ingest_equipment(420.0, 40.7132, -74.0058, "P-100", "pole", 0.8)

    # A second report at the same point but outside the clustering window
    # lands in its own event, then gets merged by an operator.
    second = board.ingest_report(7200.0, 40.7130, -74.0055, "downed_wire", "public")

    board.dispatch(first.event_id, "crew-1", 800.0)
    duplicate_blocked = False
    try:
        board.dispatch(first.event_id, "crew-2", 810.0)
    except DuplicateDispatchError:
        duplicate_blocked = True

    merge = board.merge_events([first.event_id, second.event_id], 900.0)

    return {
        "active_events": len(board.events()),
        "closed_events": len(board.events(include_closed=True)),
        "raw_records": len(board.raw_records()),
        "work_orders": len(board.work_orders()),
        "duplicate_dispatch_blocked": duplicate_blocked,
        "merged_event_id": merge.event_id,
        "merged_away": len(merge.moved_wo_ids),
        "cancelled_duplicate_orders": len(merge.cancelled_wo_ids),
        "confidence_trace_entries": len(board.confidence_trace(first.event_id)),
        "confidence": round(board.event(first.event_id).confidence, 4),
        "project": "RelayGrid",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relaygrid")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("demo", help="run a deterministic feeder demo")
    subcommands.add_parser(
        "damage-demo", help="run a deterministic damage coordination board demo"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        print(json.dumps(demo_summary(), sort_keys=True))
        return 0
    if args.command == "damage-demo":
        print(json.dumps(damage_demo_summary(), sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
