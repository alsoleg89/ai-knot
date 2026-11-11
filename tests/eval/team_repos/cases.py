"""Sealed, authored repository-maintenance cases; no provider or runtime dependency."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Document:
    path: str
    text: str
    task_key: str


@dataclass(frozen=True)
class Stage:
    id: str
    brief: str
    documents: tuple[Document, ...]
    test_source: str
    reference_solution: str


@dataclass(frozen=True)
class Case:
    id: str
    task_key: str
    title: str
    goal: str
    requirements: tuple[str, ...]
    initial_code: str
    stages: tuple[Stage, ...]


CASES: tuple[Case, ...] = (
    Case(
        id=("parcel-ledger"),
        task_key="parcel-flow",
        title=("Repair the parcel dispatch ledger"),
        goal=(
            "Implement a pure parcel-event batch router and maintain it across a policy revision."
        ),
        requirements=(
            "Only solution.py is editable; keep route_events(events, seen=()).",
            "Use the dated parcel-flow sources for the stage; other task labels "
            "are separate clients.",
            "Use only Python standard-library code and do not mutate caller-owned data.",
        ),
        initial_code=(
            'def route_events(events, seen=()):\n    """Return (dispatch'
            'es, sorted seen IDs) for parcel events without mutation.""'
            '"\n    return [], list(seen)\n'
        ),
        stages=(
            Stage(
                id=("revision-1"),
                brief=(
                    "Repair solution.py for parcel-flow. Implement the batch ro"
                    "uting and deduplication\ncontract in the October 26 revisio"
                    "n-1 documents. Keep its public signature and\nreturn shape;"
                    " make the offline checks pass.\n"
                ),
                documents=(
                    Document(
                        path=("parcel/operations.md"),
                        text=(
                            "Task parcel-flow, operations log, 2026-10-26, routing poli"
                            "cy revision 1.\nFor each unseen event, trim and casefold it"
                            "s zone. north uses N1 and south\nuses S1 through 1000 grams"
                            " inclusive; heavier known-zone parcels use BULK.\nUnknown z"
                            "ones dispatch to HOLD regardless of weight. Every dispatch"
                            ",\nincluding HOLD, records its event_id. Process the batch "
                            "in input order.\n"
                        ),
                        task_key=("parcel-flow"),
                    ),
                    Document(
                        path=("parcel/integration.md"),
                        text=(
                            "Task parcel-flow, API v1, 2026-10-26. Keep route_events(ev"
                            "ents, seen=()).\nEach event has event_id (nonempty string),"
                            " zone (string), grams (positive\ninteger); inputs are alrea"
                            "dy structurally valid. Return a pair: a list of\n(event_id,"
                            " lane) tuples and a sorted list of all previously/newly re"
                            "corded\nIDs. Seen may be a list or tuple with duplicates. C"
                            "opy; never mutate events,\nevent dictionaries, or seen. IDs"
                            " are exact, case sensitive strings, untrimmed.\n"
                        ),
                        task_key=("parcel-flow"),
                    ),
                    Document(
                        path=("parcel/shift-log.md"),
                        text=(
                            "Task parcel-flow, shift handoff, 2026-10-26. Suppress an e"
                            "vent whose exact ID\nis already recorded, before considerin"
                            "g its zone or weight. Only the first\noccurrence of a new I"
                            "D dispatches in a batch. Empty input returns no\ndispatches"
                            " and the sorted unique seen IDs. Sorting uses Python strin"
                            "g order.\nA HOLD can therefore block a corrected resend und"
                            "er revision 1.\n"
                        ),
                        task_key=("parcel-flow"),
                    ),
                    Document(
                        path=("parcel/labels.md"),
                        text=(
                            "Task parcel-flow, label desk, 2026-10-26. The browser prin"
                            "ts the destination\nzone in title case and keeps the entere"
                            "d weight for the receipt. The yellow\nlabel stock has moved"
                            " to shelf four. These presentation changes do not add\nrece"
                            "ipt fields or printed text to route_events' result.\n"
                        ),
                        task_key=("parcel-flow"),
                    ),
                    Document(
                        path=("team/working-agreements.md"),
                        text=(
                            "Shared team note, 2026-10-28. Task labels identify separat"
                            "e maintenance streams.\nA note for another task describes t"
                            "hat team's caller; similar function names do\nnot change th"
                            "is repository's contract. Dated revisions apply to the nam"
                            "ed task.\nPatch solution.py using Python's standard library"
                            ". Our CI runs entirely offline.\n"
                        ),
                        task_key=(""),
                    ),
                    Document(
                        path=("stock/shift-log.md"),
                        text=(
                            "Task stock-rebalance, 2026-10-26. Our separate bin-transfe"
                            "r planner calls a\nnorth aisle N2, uses BULK only above 200"
                            "0 grams, and does not remember HOLD.\nInventory transfer ti"
                            "ckets are not parcel dispatch events.\n"
                        ),
                        task_key=("stock-rebalance"),
                    ),
                ),
                test_source=(
                    "import copy\nimport unittest\nfrom solution import route_eve"
                    "nts\n\nclass ParcelChecks(unittest.TestCase):\n    def test_r"
                    "outing_and_boundary(self):\n        rows = [dict(event_id='"
                    "a', zone=' north ', grams=1),\n                dict(event_i"
                    "d='b', zone='SOUTH', grams=1000),\n                dict(eve"
                    "nt_id='c', zone='north', grams=1001),\n                dict"
                    "(event_id='d', zone='west', grams=9999)]\n        self.asse"
                    "rtEqual(route_events(rows),\n                         ([('a"
                    "', 'N1'), ('b', 'S1'), ('c', 'BULK'), ('d', 'HOLD')],\n    "
                    "                      ['a', 'b', 'c', 'd']))\n        self."
                    "assertEqual(route_events([dict(event_id='z', zone='north',"
                    " grams=1000)]),\n                         ([('z', 'N1')], ["
                    "'z']))\n        self.assertEqual(route_events([dict(event_i"
                    "d='z', zone='', grams=1)]),\n                         ([('z"
                    "', 'HOLD')], ['z']))\n    def test_identity_and_purity(self"
                    "):\n        rows = [dict(event_id='q', zone='west', grams=1"
                    "),\n                dict(event_id='q', zone='north', grams="
                    "1),\n                dict(event_id='Q', zone='south', grams"
                    "=1),\n                dict(event_id=' q ', zone='south', gr"
                    "ams=1)]\n        original = copy.deepcopy(rows)\n        see"
                    "n = ['old', 'old']\n        self.assertEqual(route_events(r"
                    "ows, seen),\n                         ([('q', 'HOLD'), ('Q'"
                    ", 'S1'), (' q ', 'S1')],\n                          [' q ',"
                    " 'Q', 'old', 'q']))\n        self.assertEqual(rows, origina"
                    "l)\n        self.assertEqual(seen, ['old', 'old'])\n        "
                    "self.assertEqual(route_events(rows, ['q', 'Q', ' q ']),\n  "
                    "                       ([], [' q ', 'Q', 'q']))\n        se"
                    "lf.assertEqual(route_events([], ('b', 'a', 'b')), ([], ['a"
                    "', 'b']))\n        self.assertEqual(route_events([]), ([], "
                    "[]))\n        self.assertEqual(route_events([dict(event_id="
                    "'m', zone=' SOUTH ', grams=1001)]),\n                      "
                    "   ([('m', 'BULK')], ['m']))\n"
                ),
                reference_solution=(
                    'def route_events(events, seen=()):\n    """Return (dispatch'
                    'es, sorted seen IDs) for parcel events without mutation.""'
                    '"\n    known = set(seen)\n    dispatches = []\n    for event '
                    "in events:\n        key = event['event_id']\n        if key "
                    "in known:\n            continue\n        zone = event['zone'"
                    "].strip().casefold()\n        if zone not in ('north', 'sou"
                    "th'):\n            lane = 'HOLD'\n        elif event['grams'"
                    "] > 1000:\n            lane = 'BULK'\n        else:\n        "
                    "    lane = 'N1' if zone == 'north' else 'S1'\n        dispa"
                    "tches.append((key, lane))\n        known.add(key)\n    retur"
                    "n dispatches, sorted(known)\n"
                ),
            ),
            Stage(
                id=("revision-2"),
                brief=(
                    "Update the same solution.py for parcel-flow's October 28 r"
                    "evision 2. The operations\nand handoff records were replace"
                    "d: apply the new weight boundary and deferral\nbehavior whi"
                    "le preserving API v1 and the unaffected contract. Concurre"
                    "nt notes\nwith other task labels describe their own clients"
                    ".\n"
                ),
                documents=(
                    Document(
                        path=("parcel/operations.md"),
                        text=(
                            "Task parcel-flow, operations log, 2026-10-28, routing poli"
                            "cy revision 2.\nThis replaces revision 1 from October 26. K"
                            "nown north/south zones still\ntrim/casefold and use N1/S1, "
                            "but the small-parcel range is now strictly below\n1000 gram"
                            "s: exactly 1000 is BULK. Unknown-zone events are deferred:"
                            " emit no\ndispatch and do not record their IDs. A later cor"
                            "rected event can dispatch.\n"
                        ),
                        task_key=("parcel-flow"),
                    ),
                    Document(
                        path=("parcel/integration.md"),
                        text=(
                            "Task parcel-flow, API v1, 2026-10-28. Signature and return"
                            " pair remain\nroute_events(events, seen=()) -> (list of (ev"
                            "ent_id, lane), sorted unique IDs).\nEvents have nonempty st"
                            "ring event_id, string zone, positive integer grams.\nInputs"
                            " are valid and must not be mutated. IDs are exact case sen"
                            "sitive\nuntrimmed strings. Previously seen IDs remain recor"
                            "ded even for unknown zones.\n"
                        ),
                        task_key=("parcel-flow"),
                    ),
                    Document(
                        path=("parcel/shift-log.md"),
                        text=(
                            "Task parcel-flow, shift handoff, 2026-10-28. For revision "
                            "2, suppress IDs\nalready recorded before checking the event"
                            ". Iterate in batch order and add\nan ID only after a dispat"
                            "ch. Several deferred records of the same ID remain\nunrecor"
                            "ded; the first later known-zone record dispatches and supp"
                            "resses\nsubsequent duplicates. Empty batches preserve sorte"
                            "d unique seen IDs.\n"
                        ),
                        task_key=("parcel-flow"),
                    ),
                    Document(
                        path=("parcel/labels.md"),
                        text=(
                            "Task parcel-flow, label desk, 2026-10-26. The browser prin"
                            "ts the destination\nzone in title case and keeps the entere"
                            "d weight for the receipt. The yellow\nlabel stock has moved"
                            " to shelf four. These presentation changes do not add\nrece"
                            "ipt fields or printed text to route_events' result.\n"
                        ),
                        task_key=("parcel-flow"),
                    ),
                    Document(
                        path=("team/working-agreements.md"),
                        text=(
                            "Shared team note, 2026-10-28. Task labels identify separat"
                            "e maintenance streams.\nA note for another task describes t"
                            "hat team's caller; similar function names do\nnot change th"
                            "is repository's contract. Dated revisions apply to the nam"
                            "ed task.\nPatch solution.py using Python's standard library"
                            ". Our CI runs entirely offline.\n"
                        ),
                        task_key=(""),
                    ),
                    Document(
                        path=("stock/shift-log.md"),
                        text=(
                            "Task stock-rebalance, 2026-10-28, inventory planner revisi"
                            "on 8. Keep our 2000\ngram threshold inclusive and use N2 fo"
                            "r north aisles. Record HOLD tickets\nto prevent repeated in"
                            "ventory bin moves. This rollout is for stock transfers.\n"
                        ),
                        task_key=("stock-rebalance"),
                    ),
                    Document(
                        path=("pickup/planning.md"),
                        text=(
                            "Task pickup-slots, 2026-10-28. For van appointment plannin"
                            "g an unrecognised\ndistrict must still reserve a slot named"
                            " HOLD. District strings are stored\nas entered. The appoint"
                            "ment planner has a different identifier ledger.\n"
                        ),
                        task_key=("pickup-slots"),
                    ),
                ),
                test_source=(
                    "import copy\nimport unittest\nfrom solution import route_eve"
                    "nts\n\nclass ParcelChecks(unittest.TestCase):\n    def test_c"
                    "urrent_policy(self):\n        rows = [dict(event_id='a', zo"
                    "ne=' north ', grams=999),\n                dict(event_id='b"
                    "', zone='SOUTH', grams=1000),\n                dict(event_i"
                    "d='c', zone='north', grams=1001),\n                dict(eve"
                    "nt_id='d', zone='west', grams=9999)]\n        self.assertEq"
                    "ual(route_events(rows),\n                         ([('a', '"
                    "N1'), ('b', 'BULK'), ('c', 'BULK')], ['a', 'b', 'c']))\n   "
                    "     self.assertEqual(route_events([dict(event_id='z', zon"
                    "e='north', grams=1000)]),\n                         ([('z',"
                    " 'BULK')], ['z']))\n        self.assertEqual(route_events(["
                    "dict(event_id='z', zone='', grams=1)]), ([], []))\n        "
                    "self.assertEqual(route_events([dict(event_id='s', zone='so"
                    "uth', grams=1)]),\n                         ([('s', 'S1')],"
                    " ['s']))\n    def test_deferral_identity_and_purity(self):\n"
                    "        rows = [dict(event_id='q', zone='west', grams=1),\n"
                    "                dict(event_id='q', zone='', grams=1),\n    "
                    "            dict(event_id='q', zone='north', grams=1),\n   "
                    "             dict(event_id='q', zone='south', grams=1),\n  "
                    "              dict(event_id='Q', zone='south', grams=1),\n "
                    "               dict(event_id=' q ', zone='south', grams=1)"
                    "]\n        original = copy.deepcopy(rows)\n        seen = ['"
                    "old', 'old']\n        self.assertEqual(route_events(rows, s"
                    "een),\n                         ([('q', 'N1'), ('Q', 'S1'),"
                    " (' q ', 'S1')],\n                          [' q ', 'Q', 'o"
                    "ld', 'q']))\n        self.assertEqual(rows, original)\n     "
                    "   self.assertEqual(seen, ['old', 'old'])\n        self.ass"
                    "ertEqual(route_events(rows, ['q', 'Q', ' q ']),\n          "
                    "               ([], [' q ', 'Q', 'q']))\n        self.asser"
                    "tEqual(route_events([], ('b', 'a', 'b')), ([], ['a', 'b'])"
                    ")\n        self.assertEqual(route_events([]), ([], []))\n   "
                    "     self.assertEqual(route_events([dict(event_id='x', zon"
                    "e='unknown', grams=1)], ['x']),\n                         ("
                    "[], ['x']))\n"
                ),
                reference_solution=(
                    'def route_events(events, seen=()):\n    """Return (dispatch'
                    'es, sorted seen IDs) for parcel events without mutation.""'
                    '"\n    known = set(seen)\n    dispatches = []\n    for event '
                    "in events:\n        key = event['event_id']\n        if key "
                    "in known:\n            continue\n        zone = event['zone'"
                    "].strip().casefold()\n        if zone not in ('north', 'sou"
                    "th'):\n            continue\n        lane = 'BULK' if event["
                    "'grams'] >= 1000 else ('N1' if zone == 'north' else 'S1')\n"
                    "        dispatches.append((key, lane))\n        known.add(k"
                    "ey)\n    return dispatches, sorted(known)\n"
                ),
            ),
            Stage(
                id=("revision-3"),
                brief=(
                    "Maintain parcel-flow under the October 30 revision 3. Appl"
                    "y the new south-lane weight boundary and accepted short zo"
                    "ne names from the replacement sources, preserving API v1, "
                    "the north boundary and the idempotency/deferral contract.\n"
                ),
                documents=(
                    Document(
                        path=("parcel/operations.md"),
                        text=(
                            "Task parcel-flow, operations log, 2026-10-30, routing poli"
                            "cy revision 3.\nReplaces revision 2 of October 28. North st"
                            "ays N1 strictly below 1000 grams;\nsouth now uses S1 throug"
                            "h 1500 grams inclusive. Heavier parcels use BULK.\nAfter st"
                            "rip/casefold, accept n as north and s as south as well as "
                            "full names.\nAll other zones remain deferred with no dispat"
                            "ch and no ID recorded.\n"
                        ),
                        task_key=("parcel-flow"),
                    ),
                    Document(
                        path=("parcel/integration.md"),
                        text=(
                            "Task parcel-flow, API v1, 2026-10-30. Signature stays rout"
                            "e_events(events,\nseen=()). Return (list of (event_id, lane"
                            ") tuples in batch order, sorted\nunique recorded IDs). Ever"
                            "y event has nonempty string event_id, string zone\nand posi"
                            "tive integer grams. Inputs are valid. Copy caller data; do"
                            " not\nmutate events, dictionaries or seen. IDs are exact ca"
                            "se sensitive strings.\n"
                        ),
                        task_key=("parcel-flow"),
                    ),
                    Document(
                        path=("parcel/shift-log.md"),
                        text=(
                            "Task parcel-flow, shift handoff, 2026-10-30. Suppress reco"
                            "rded IDs before\nrouting, and record an ID only after a dis"
                            "patch. Deferral leaves the ID\navailable for a later correc"
                            "tion in the same batch. First dispatch wins;\nlater duplica"
                            "tes are suppressed. Preserve previously seen IDs even when"
                            " a\nmatching row is deferred. Empty input returns sorted un"
                            "ique seen IDs.\n"
                        ),
                        task_key=("parcel-flow"),
                    ),
                    Document(
                        path=("parcel/labels.md"),
                        text=(
                            "Task parcel-flow, label desk, 2026-10-26. The browser prin"
                            "ts the destination\nzone in title case and keeps the entere"
                            "d weight for the receipt. The yellow\nlabel stock has moved"
                            " to shelf four. These presentation changes do not add\nrece"
                            "ipt fields or printed text to route_events' result.\n"
                        ),
                        task_key=("parcel-flow"),
                    ),
                    Document(
                        path=("team/working-agreements.md"),
                        text=(
                            "Shared team note, 2026-10-28. Task labels identify separat"
                            "e maintenance streams.\nA note for another task describes t"
                            "hat team's caller; similar function names do\nnot change th"
                            "is repository's contract. Dated revisions apply to the nam"
                            "ed task.\nPatch solution.py using Python's standard library"
                            ". Our CI runs entirely offline.\n"
                        ),
                        task_key=(""),
                    ),
                    Document(
                        path=("stock/shift-log.md"),
                        text=(
                            "Task stock-rebalance, 2026-10-30. Inventory moves still us"
                            "e N2, and the\nsouth aisle now has a 1200 gram inclusive th"
                            "reshold. Short aisle names n\nand s are not accepted in the"
                            " inventory import stream.\n"
                        ),
                        task_key=("stock-rebalance"),
                    ),
                    Document(
                        path=("pickup/planning.md"),
                        text=(
                            "Task pickup-slots, 2026-10-30. Appointment IDs ignore lett"
                            "er case to merge\nrescheduled slots. Any unknown district r"
                            "eserves HOLD immediately. North\nand south both have a 1500"
                            " gram exclusive planning boundary.\n"
                        ),
                        task_key=("pickup-slots"),
                    ),
                ),
                test_source=(
                    "import copy\nimport unittest\nfrom solution import route_eve"
                    "nts\n\nclass ParcelChecks(unittest.TestCase):\n    def test_c"
                    "urrent_zones_and_limits(self):\n        self.assertEqual(ro"
                    "ute_events([dict(event_id='a', zone='north', grams=999)]),"
                    "\n                         ([('a', 'N1')], ['a']))\n        "
                    "self.assertEqual(route_events([dict(event_id='a', zone='no"
                    "rth', grams=1000)]),\n                         ([('a', 'BUL"
                    "K')], ['a']))\n        self.assertEqual(route_events([dict("
                    "event_id='a', zone='south', grams=1000)]),\n               "
                    "          ([('a', 'S1')], ['a']))\n        self.assertEqual"
                    "(route_events([dict(event_id='a', zone='south', grams=1500"
                    ")]),\n                         ([('a', 'S1')], ['a']))\n    "
                    "    self.assertEqual(route_events([dict(event_id='a', zone"
                    "='south', grams=1501)]),\n                         ([('a', "
                    "'BULK')], ['a']))\n        self.assertEqual(route_events([d"
                    "ict(event_id='a', zone=' N ', grams=1),\n                  "
                    "                     dict(event_id='b', zone=' S ', grams="
                    "1500)]),\n                         ([('a', 'N1'), ('b', 'S1"
                    "')], ['a', 'b']))\n    def test_ledger_and_purity(self):\n  "
                    "      rows = [dict(event_id='q', zone='west', grams=1),\n  "
                    "              dict(event_id='q', zone='s', grams=1200),\n  "
                    "              dict(event_id='q', zone='north', grams=1),\n "
                    "               dict(event_id='Q', zone='n', grams=1),\n    "
                    "            dict(event_id=' q ', zone='south', grams=1)]\n "
                    "       original = copy.deepcopy(rows)\n        seen = ['z',"
                    " 'z']\n        self.assertEqual(route_events(rows, seen),\n "
                    "                        ([('q', 'S1'), ('Q', 'N1'), (' q '"
                    ", 'S1')],\n                          [' q ', 'Q', 'q', 'z']"
                    "))\n        self.assertEqual(rows, original)\n        self.a"
                    "ssertEqual(seen, ['z', 'z'])\n        self.assertEqual(rout"
                    "e_events([], ('z', 'a', 'z')), ([], ['a', 'z']))\n        s"
                    "elf.assertEqual(route_events([dict(event_id='a', zone='?',"
                    " grams=1)]), ([], []))\n        self.assertEqual(route_even"
                    "ts([dict(event_id='a', zone='s', grams=1)], ['a']),\n      "
                    "                   ([], ['a']))\n"
                ),
                reference_solution=(
                    'def route_events(events, seen=()):\n    """Return (dispatch'
                    'es, sorted seen IDs) for parcel events without mutation.""'
                    '"\n    known = set(seen)\n    dispatches = []\n    for event '
                    "in events:\n        key = event['event_id']\n        if key "
                    "in known:\n            continue\n        zone = event['zone'"
                    "].strip().casefold()\n        zone = {'n': 'north', 's': 's"
                    "outh'}.get(zone, zone)\n        if zone not in ('north', 's"
                    "outh'):\n            continue\n        small = event['grams'"
                    "] < 1000 if zone == 'north' else event['grams'] <= 1500\n  "
                    "      lane = ('N1' if zone == 'north' else 'S1') if small "
                    "else 'BULK'\n        dispatches.append((key, lane))\n       "
                    " known.add(key)\n    return dispatches, sorted(known)\n"
                ),
            ),
        ),
    ),
    Case(
        id=("notes-digest"),
        task_key="notes-export",
        title=("Maintain the completed-note Markdown exporter"),
        goal=(
            "Produce a deterministic Markdown digest from structured no"
            "tes as the export policy changes."
        ),
        requirements=(
            "Only solution.py is editable; keep render_digest(rows, max_items=2).",
            "Use notes-export documents for selection, grouping and Markdown formatting.",
            "Return a string with deterministic order using standard-library "
            "Python; preserve inputs.",
        ),
        initial_code=(
            'def render_digest(rows, max_items=2):\n    """Render a comp'
            'leted-note digest as Markdown without changing rows."""\n  '
            "  return '# Digest\\n'\n"
        ),
        stages=(
            Stage(
                id=("revision-1"),
                brief=(
                    "Исправь solution.py: реализуй render_digest для задачи not"
                    "es-export по документам\nот 26 октября (revision 1). Нужен "
                    "полный код функции, сохраняющий API и точный\nMarkdown-форм"
                    "ат. Документы редакторского preview относятся к другой зад"
                    "аче.\n"
                ),
                documents=(
                    Document(
                        path=("notes/export.md"),
                        text=(
                            "Task notes-export, Markdown export revision 1, 2026-10-26."
                            " Select only rows\nwhose status is exactly 'done'. Collapse"
                            " all whitespace runs in team and text\nto single spaces and"
                            " trim both ends, using Python split/join semantics. Drop\nr"
                            "ows with empty normalized text. An empty team becomes Gene"
                            "ral. Group by\nnormalized team exactly, case sensitively; r"
                            "etain duplicate note texts.\n"
                        ),
                        task_key=("notes-export"),
                    ),
                    Document(
                        path=("notes/formatting.md"),
                        text=(
                            "Task notes-export, 2026-10-26. Start with '# Digest'. For "
                            "each team, append a\nblank line, then '## ' plus team, then"
                            " one '- ' plus note per visible note.\nSort teams by (team."
                            "casefold(), team); notes retain input order in each team.\n"
                            "Show at most max_items notes per team. If n notes are hidd"
                            "en append exactly\n'- ... (+n more)'. Count duplicates in n"
                            ". Always end with one newline; with\nno selected notes the "
                            "complete output is '# Digest\\n'. Do not escape punctuation"
                            ".\n"
                        ),
                        task_key=("notes-export"),
                    ),
                    Document(
                        path=("notes/adapter.md"),
                        text=(
                            "Task notes-export, API v1, 2026-10-26. Keep render_digest("
                            "rows, max_items=2).\nrows is a list of dictionaries with st"
                            "ring team, text and status; max_items\nis a nonnegative int"
                            "eger. With zero, still show team headers and the hidden\nco"
                            "unt line for every nonempty group. This is a pure string t"
                            "ransformation:\nnever mutate rows or their dictionaries, an"
                            "d do not add a timestamp.\n"
                        ),
                        task_key=("notes-export"),
                    ),
                    Document(
                        path=("notes/review.md"),
                        text=(
                            "Task notes-export, editor meeting, 2026-10-26. The desktop"
                            " viewer will display\nthe digest in a narrow panel, and the"
                            " product team prefers sentence case in\nauthor-written note"
                            "s. The viewer owns colors and fonts. This meeting did not\n"
                            "approve a change to selection rules or Markdown punctuatio"
                            "n.\n"
                        ),
                        task_key=("notes-export"),
                    ),
                    Document(
                        path=("team/working-agreements.md"),
                        text=(
                            "Shared team note, 2026-10-28. Task labels identify separat"
                            "e maintenance streams.\nA note for another task describes t"
                            "hat team's caller; similar function names do\nnot change th"
                            "is repository's contract. Dated revisions apply to the nam"
                            "ed task.\nPatch solution.py using Python's standard library"
                            ". Our CI runs entirely offline.\n"
                        ),
                        task_key=(""),
                    ),
                    Document(
                        path=("preview/formatting.md"),
                        text=(
                            "Task editor-preview, 2026-10-26. Our live preview groups t"
                            "eams ignoring case,\nremoves repeated text and includes clo"
                            "sed notes. Preview rows never leave\nthe browser. This sepa"
                            "rate feature has not changed notes-export revision 1.\n"
                        ),
                        task_key=("editor-preview"),
                    ),
                ),
                test_source=(
                    "import copy\nimport unittest\nfrom solution import render_di"
                    "gest\n\nclass DigestChecks(unittest.TestCase):\n    def test_"
                    "selection_and_layout(self):\n        self.assertEqual(rende"
                    "r_digest([]), '# Digest\\n')\n        self.assertEqual(rende"
                    "r_digest([dict(team='A', text='x', status='closed')]), '# "
                    "Digest\\n')\n        self.assertEqual(render_digest([dict(te"
                    "am='A', text='x', status=' DONE ')]), '# Digest\\n')\n      "
                    "  self.assertEqual(render_digest([dict(team=' A  B ', text"
                    "=' hi\\nthere ', status='done')]),\n                        "
                    " '# Digest\\n\\n## A B\\n- hi there\\n')\n        self.assertEq"
                    "ual(render_digest([dict(team=' ', text='  ', status='done'"
                    ")]), '# Digest\\n')\n        self.assertEqual(render_digest("
                    "[dict(team='', text='a|b *c*', status='done')]),\n         "
                    "                '# Digest\\n\\n## General\\n- a|b *c*\\n')\n   "
                    " def test_grouping_limits_and_purity(self):\n        rows ="
                    " [dict(team='z', text='last', status='done'),\n            "
                    "    dict(team='a', text='first', status='done'),\n         "
                    "       dict(team='A', text='other', status='done'),\n      "
                    "          dict(team='a', text='first', status='done'),\n   "
                    "             dict(team='a', text='third', status='done')]\n"
                    "        original = copy.deepcopy(rows)\n        self.assert"
                    "Equal(render_digest(rows),\n                         '# Dig"
                    "est\\n\\n## A\\n- other\\n\\n## a\\n- first\\n- first\\n- ... (+1 "
                    "more)\\n\\n## z\\n- last\\n')\n        self.assertEqual(render_"
                    "digest(rows, 0),\n                         '# Digest\\n\\n## "
                    "A\\n- ... (+1 more)\\n\\n## a\\n- ... (+3 more)\\n\\n## z\\n- ..."
                    " (+1 more)\\n')\n        self.assertEqual(render_digest([dic"
                    "t(team='x', text='n', status='done')], 1),\n               "
                    "          '# Digest\\n\\n## x\\n- n\\n')\n        self.assertEq"
                    "ual(render_digest([dict(team='Саппорт', text='готово', sta"
                    "tus='done')]),\n                         '# Digest\\n\\n## Са"
                    "ппорт\\n- готово\\n')\n        self.assertEqual(rows, origina"
                    "l)\n"
                ),
                reference_solution=(
                    'def render_digest(rows, max_items=2):\n    """Render a comp'
                    'leted-note digest as Markdown without changing rows."""\n  '
                    "  groups = {}\n    for row in rows:\n        if row['status'"
                    "] != 'done':\n            continue\n        text = ' '.join("
                    "row['text'].split())\n        if not text:\n            cont"
                    "inue\n        team = ' '.join(row['team'].split()) or 'Gene"
                    "ral'\n        groups.setdefault(team, []).append(text)\n    "
                    "lines = ['# Digest']\n    for team in sorted(groups, key=la"
                    "mbda value: (value.casefold(), value)):\n        items = gr"
                    "oups[team]\n        lines.extend(['', '## ' + team])\n      "
                    "  lines.extend('- ' + text for text in items[:max_items])\n"
                    "        if len(items) > max_items:\n            lines.appen"
                    "d(f'- ... (+{len(items) - max_items} more)')\n    return '\\"
                    "n'.join(lines) + '\\n'\n"
                ),
            ),
            Stage(
                id=("revision-2"),
                brief=(
                    "Обнови solution.py по новой редакции notes-export от 28 ок"
                    "тября: изменились\nдопустимые статусы и правила объединения"
                    " групп/повторов до усечения. Сохрани\nподпись функции, поря"
                    "док оставшихся заметок и описанный Markdown-формат.\nСтарые"
                    " записи от 26 октября больше не задают эти изменённые прав"
                    "ила.\n"
                ),
                documents=(
                    Document(
                        path=("notes/export.md"),
                        text=(
                            "Task notes-export, Markdown export revision 2, 2026-10-28."
                            " Replaces the\nOctober 26 rules. Normalize status by strip/"
                            "casefold; select done and closed.\nTeam/text whitespace nor"
                            "malization and dropping empty text are unchanged.\nGroup te"
                            "ams by casefolded normalized name, displaying the first se"
                            "lected\nnonblank-text row's normalized team spelling (blank"
                            " still means General).\nWithin each group keep only the fir"
                            "st text for each casefolded normalized\ntext. This deduplic"
                            "ation happens before truncation and hidden-count calculati"
                            "on.\n"
                        ),
                        task_key=("notes-export"),
                    ),
                    Document(
                        path=("notes/formatting.md"),
                        text=(
                            "Task notes-export, 2026-10-28. Format remains '# Digest', "
                            "then a blank line,\n'## ' plus displayed team and '- ' plus"
                            " each visible note. Sort groups by\ncasefolded name; preser"
                            "ve first occurrence order of unique notes. max_items\nlimit"
                            "s each group's unique notes. Hidden notes use '- ... (+n m"
                            "ore)'. No\nescaping, timestamps or trailing spaces. End wit"
                            "h one newline. No selected\nnotes yields exactly '# Digest\\"
                            "n'. This is revision 2, not the live preview.\n"
                        ),
                        task_key=("notes-export"),
                    ),
                    Document(
                        path=("notes/adapter.md"),
                        text=(
                            "Task notes-export, API v1, 2026-10-28. Keep render_digest("
                            "rows, max_items=2),\nreturning a string. Every row has stri"
                            "ng team/text/status. max_items is a\nnonnegative integer; z"
                            "ero leaves team headers plus hidden unique-note counts.\nDo"
                            " not mutate input data. The result retains the first note'"
                            "s spelling even\nwhen a later spelling is different only in"
                            " letter case.\n"
                        ),
                        task_key=("notes-export"),
                    ),
                    Document(
                        path=("notes/review.md"),
                        text=(
                            "Task notes-export, editor meeting, 2026-10-26. The desktop"
                            " viewer will display\nthe digest in a narrow panel, and the"
                            " product team prefers sentence case in\nauthor-written note"
                            "s. The viewer owns colors and fonts. This meeting did not\n"
                            "approve a change to selection rules or Markdown punctuatio"
                            "n.\n"
                        ),
                        task_key=("notes-export"),
                    ),
                    Document(
                        path=("team/working-agreements.md"),
                        text=(
                            "Shared team note, 2026-10-28. Task labels identify separat"
                            "e maintenance streams.\nA note for another task describes t"
                            "hat team's caller; similar function names do\nnot change th"
                            "is repository's contract. Dated revisions apply to the nam"
                            "ed task.\nPatch solution.py using Python's standard library"
                            ". Our CI runs entirely offline.\n"
                        ),
                        task_key=(""),
                    ),
                    Document(
                        path=("preview/formatting.md"),
                        text=(
                            "Task editor-preview, 2026-10-28. In the separate live prev"
                            "iew, include every\nstatus except deleted, display the most"
                            " recently entered team spelling, and\nshow three notes by d"
                            "efault. Keep repeated messages so editors see each row.\n"
                        ),
                        task_key=("editor-preview"),
                    ),
                    Document(
                        path=("mail/adapter.md"),
                        text=(
                            "Task mail-roundup, 2026-10-28. The outbound plain-text rou"
                            "ndup sorts notes\nalphabetically, omits headings when its l"
                            "imit is zero, and uses an ASCII\ntitle. This adapter is mai"
                            "ntained separately from the Markdown digest.\n"
                        ),
                        task_key=("mail-roundup"),
                    ),
                ),
                test_source=(
                    "import copy\nimport unittest\nfrom solution import render_di"
                    "gest\n\nclass DigestChecks(unittest.TestCase):\n    def test_"
                    "current_selection_and_layout(self):\n        self.assertEqu"
                    "al(render_digest([]), '# Digest\\n')\n        self.assertEqu"
                    "al(render_digest([dict(team='A', text='x', status=' closed"
                    " ')]),\n                         '# Digest\\n\\n## A\\n- x\\n')"
                    "\n        self.assertEqual(render_digest([dict(team='A', te"
                    "xt='x', status=' DONE ')]),\n                         '# Di"
                    "gest\\n\\n## A\\n- x\\n')\n        self.assertEqual(render_dige"
                    "st([dict(team='A', text='x', status='open')]), '# Digest\\n"
                    "')\n        self.assertEqual(render_digest([dict(team=' A  "
                    "B ', text=' hi\\nthere ', status='done')]),\n               "
                    "          '# Digest\\n\\n## A B\\n- hi there\\n')\n        self"
                    ".assertEqual(render_digest([dict(team=' ', text='  ', stat"
                    "us='done')]), '# Digest\\n')\n        self.assertEqual(rende"
                    "r_digest([dict(team='', text='a|b *c*', status='done')]),\n"
                    "                         '# Digest\\n\\n## General\\n- a|b *c"
                    "*\\n')\n    def test_grouping_limits_and_purity(self):\n     "
                    "   rows = [dict(team='z', text='last', status='done'),\n   "
                    "             dict(team='a', text='First', status='done'),\n"
                    "                dict(team='A', text='first', status='close"
                    "d'),\n                dict(team='a', text='second', status="
                    "'done'),\n                dict(team='a', text='third', stat"
                    "us='done')]\n        original = copy.deepcopy(rows)\n       "
                    " self.assertEqual(render_digest(rows),\n                   "
                    "      '# Digest\\n\\n## a\\n- First\\n- second\\n- ... (+1 more"
                    ")\\n\\n## z\\n- last\\n')\n        self.assertEqual(render_dige"
                    "st(rows, 0),\n                         '# Digest\\n\\n## a\\n-"
                    " ... (+3 more)\\n\\n## z\\n- ... (+1 more)\\n')\n        self.a"
                    "ssertEqual(render_digest([dict(team='Саппорт', text='ГОТОВ"
                    "О', status='done'),\n                                      "
                    "  dict(team='саппорт', text='готово', status='closed')]),\n"
                    "                         '# Digest\\n\\n## Саппорт\\n- ГОТОВО"
                    "\\n')\n        self.assertEqual(rows, original)\n"
                ),
                reference_solution=(
                    'def render_digest(rows, max_items=2):\n    """Render a comp'
                    'leted-note digest as Markdown without changing rows."""\n  '
                    "  groups = {}\n    for row in rows:\n        if row['status'"
                    "].strip().casefold() not in ('done', 'closed'):\n          "
                    "  continue\n        text = ' '.join(row['text'].split())\n  "
                    "      if not text:\n            continue\n        team = ' '"
                    ".join(row['team'].split()) or 'General'\n        name, item"
                    "s, keys = groups.setdefault(team.casefold(), (team, [], se"
                    "t()))\n        if text.casefold() not in keys:\n            "
                    "items.append(text)\n            keys.add(text.casefold())\n "
                    "   lines = ['# Digest']\n    for key in sorted(groups):\n   "
                    "     team, items, _ = groups[key]\n        lines.extend([''"
                    ", '## ' + team])\n        lines.extend('- ' + text for text"
                    " in items[:max_items])\n        if len(items) > max_items:\n"
                    "            lines.append(f'- ... (+{len(items) - max_items"
                    "} more)')\n    return '\\n'.join(lines) + '\\n'\n"
                ),
            ),
            Stage(
                id=("revision-3"),
                brief=(
                    "Поддержи notes-export по редакции от 30 октября: теперь ну"
                    "левой max_items означает отсутствие ограничения, а уникаль"
                    "ные заметки сортируются до усечения. Применяй текущие доку"
                    "менты revision 3, сохрани API v1 и остальные правила Markd"
                    "own.\n"
                ),
                documents=(
                    Document(
                        path=("notes/export.md"),
                        text=(
                            "Task notes-export, Markdown export revision 3, 2026-10-30."
                            " Replaces revision\n2 from October 28. Keep strip/casefold "
                            "status selection of done and closed.\nCollapse whitespace i"
                            "n team/text with split/join; drop empty text; empty team\nm"
                            "eans General. Merge teams by casefolded name and keep firs"
                            "t selected\nnonblank-text row's normalized team spelling. D"
                            "eduplicate notes by casefolded\nnormalized text, retaining "
                            "the first spelling of each note.\n"
                        ),
                        task_key=("notes-export"),
                    ),
                    Document(
                        path=("notes/formatting.md"),
                        text=(
                            "Task notes-export, 2026-10-30. Revision 3 sorts each group"
                            "'s unique notes by\ncasefolded text before truncation (repl"
                            "acing encounter order). Groups remain\nsorted by casefolded"
                            " name. Positive max_items caps each group; zero now means\n"
                            "unlimited, with no hidden-count line. Other formatting sta"
                            "ys '# Digest',\nblank line, '## ' plus team, '- ' plus note"
                            "s, and '- ... (+n more)' only for\nhidden notes. Keep punct"
                            "uation literal, and end with one newline. No selected\nnote"
                            "s still returns exactly '# Digest\\n'.\n"
                        ),
                        task_key=("notes-export"),
                    ),
                    Document(
                        path=("notes/adapter.md"),
                        text=(
                            "Task notes-export, API v1, 2026-10-30. Signature remains r"
                            "ender_digest(rows,\nmax_items=2), returning a deterministic"
                            " Markdown string. Rows contain string\nteam, text and statu"
                            "s; max_items is a nonnegative integer. Do not mutate rows\n"
                            "or dictionaries, add dates, or change the first preserved "
                            "spelling. The\nzero-limit interpretation and note sorting a"
                            "bove replace earlier revisions.\n"
                        ),
                        task_key=("notes-export"),
                    ),
                    Document(
                        path=("notes/review.md"),
                        text=(
                            "Task notes-export, editor meeting, 2026-10-26. The desktop"
                            " viewer will display\nthe digest in a narrow panel, and the"
                            " product team prefers sentence case in\nauthor-written note"
                            "s. The viewer owns colors and fonts. This meeting did not\n"
                            "approve a change to selection rules or Markdown punctuatio"
                            "n.\n"
                        ),
                        task_key=("notes-export"),
                    ),
                    Document(
                        path=("team/working-agreements.md"),
                        text=(
                            "Shared team note, 2026-10-28. Task labels identify separat"
                            "e maintenance streams.\nA note for another task describes t"
                            "hat team's caller; similar function names do\nnot change th"
                            "is repository's contract. Dated revisions apply to the nam"
                            "ed task.\nPatch solution.py using Python's standard library"
                            ". Our CI runs entirely offline.\n"
                        ),
                        task_key=(""),
                    ),
                    Document(
                        path=("preview/formatting.md"),
                        text=(
                            "Task editor-preview, 2026-10-30. Keep live preview notes i"
                            "n encounter order;\nzero still hides all preview items. Use"
                            " the latest team spelling so edits\nshow immediately. These"
                            " browser rules do not apply to saved Markdown export.\n"
                        ),
                        task_key=("editor-preview"),
                    ),
                    Document(
                        path=("mail/adapter.md"),
                        text=(
                            "Task mail-roundup, 2026-10-30. Sort the plain-text roundup"
                            " by the original\ncase-sensitive text and retain repeated n"
                            "otes from different authors. Its\ncompact display shows no "
                            "heading for an empty group.\n"
                        ),
                        task_key=("mail-roundup"),
                    ),
                ),
                test_source=(
                    "import copy\nimport unittest\nfrom solution import render_di"
                    "gest\n\nclass DigestChecks(unittest.TestCase):\n    def test_"
                    "current_selection_and_order(self):\n        self.assertEqua"
                    "l(render_digest([]), '# Digest\\n')\n        self.assertEqua"
                    "l(render_digest([dict(team='A', text='x', status=' CLOSED "
                    "')]),\n                         '# Digest\\n\\n## A\\n- x\\n')\n"
                    "        self.assertEqual(render_digest([dict(team='A', tex"
                    "t='x', status='open')]), '# Digest\\n')\n        self.assert"
                    "Equal(render_digest([dict(team=' A  B ', text=' hi\\nthere "
                    "', status='done')]),\n                         '# Digest\\n\\"
                    "n## A B\\n- hi there\\n')\n        self.assertEqual(render_di"
                    "gest([dict(team='', text='a|b *c*', status='done')]),\n    "
                    "                     '# Digest\\n\\n## General\\n- a|b *c*\\n'"
                    ")\n        self.assertEqual(render_digest([dict(team=' ', t"
                    "ext='  ', status='done')]), '# Digest\\n')\n    def test_lim"
                    "its_dedup_and_purity(self):\n        rows = [dict(team='z',"
                    " text='last', status='done'),\n                dict(team='a"
                    "', text='Zulu', status='done'),\n                dict(team="
                    "'A', text='alpha', status='closed'),\n                dict("
                    "team='a', text='ALPHA', status='done'),\n                di"
                    "ct(team='a', text='Beta', status='done')]\n        original"
                    " = copy.deepcopy(rows)\n        self.assertEqual(render_dig"
                    "est(rows),\n                         '# Digest\\n\\n## a\\n- a"
                    "lpha\\n- Beta\\n- ... (+1 more)\\n\\n## z\\n- last\\n')\n        "
                    "self.assertEqual(render_digest(rows, 0),\n                 "
                    "        '# Digest\\n\\n## a\\n- alpha\\n- Beta\\n- Zulu\\n\\n## z"
                    "\\n- last\\n')\n        self.assertEqual(render_digest(rows, "
                    "1),\n                         '# Digest\\n\\n## a\\n- alpha\\n-"
                    " ... (+2 more)\\n\\n## z\\n- last\\n')\n        self.assertEqua"
                    "l(render_digest([dict(team='Саппорт', text='ГОТОВО', statu"
                    "s='done'),\n                                        dict(te"
                    "am='саппорт', text='готово', status='closed')], 0),\n      "
                    "                   '# Digest\\n\\n## Саппорт\\n- ГОТОВО\\n')\n "
                    "       self.assertEqual(rows, original)\n"
                ),
                reference_solution=(
                    'def render_digest(rows, max_items=2):\n    """Render a comp'
                    'leted-note digest as Markdown without changing rows."""\n  '
                    "  groups = {}\n    for row in rows:\n        if row['status'"
                    "].strip().casefold() not in ('done', 'closed'):\n          "
                    "  continue\n        text = ' '.join(row['text'].split())\n  "
                    "      if not text:\n            continue\n        team = ' '"
                    ".join(row['team'].split()) or 'General'\n        name, item"
                    "s, keys = groups.setdefault(team.casefold(), (team, [], se"
                    "t()))\n        if text.casefold() not in keys:\n            "
                    "items.append(text)\n            keys.add(text.casefold())\n "
                    "   lines = ['# Digest']\n    for key in sorted(groups):\n   "
                    "     team, items, _ = groups[key]\n        items = sorted(i"
                    "tems, key=str.casefold)\n        limit = max_items or len(i"
                    "tems)\n        lines.extend(['', '## ' + team])\n        lin"
                    "es.extend('- ' + text for text in items[:limit])\n        i"
                    "f len(items) > limit:\n            lines.append(f'- ... (+{"
                    "len(items) - limit} more)')\n    return '\\n'.join(lines) + "
                    "'\\n'\n"
                ),
            ),
        ),
    ),
    Case(
        id=("catalog-pages"),
        task_key="catalog-client",
        title=("Correct the catalog pagination request builder"),
        goal=(
            "Build deterministic GET request data with query replacemen"
            "t, pagination and optional authentication."
        ),
        requirements=(
            "Only solution.py is editable; keep page_request(base_url, options, cursor=None).",
            "Follow the catalog-client policy revision dated for each stage.",
            "Construct request data offline with Python standard library; never "
            "send requests or mutate options.",
        ),
        initial_code=(
            'def page_request(base_url, options, cursor=None):\n    """B'
            "uild the catalog GET request dictionary without mutating o"
            "ptions.\"\"\"\n    return {'url': base_url, 'headers': {}}\n"
        ),
        stages=(
            Stage(
                id=("revision-1"),
                brief=(
                    "Fix solution.py for catalog-client using the October 26 re"
                    "quest policy revision 1.\nImplement URL/query assembly and "
                    "optional headers exactly as the API documents\ndescribe. Pr"
                    "eserve the public function and return a complete request d"
                    "ictionary.\n"
                ),
                documents=(
                    Document(
                        path=("catalog/client.md"),
                        text=(
                            "Task catalog-client, request policy revision 1, 2026-10-26"
                            ". Keep\npage_request(base_url, options, cursor=None), retur"
                            "ning {'url': string,\n'headers': dict}. This builds a GET r"
                            "equest but never sends it. page_size\ndefaults to 50; clamp"
                            " supplied integer sizes to 1..100 inclusive, so zero\nbecom"
                            "es 1. Append limit with the effective size. No need to val"
                            "idate types.\n"
                        ),
                        task_key=("catalog-client"),
                    ),
                    Document(
                        path=("catalog/transport.md"),
                        text=(
                            "Task catalog-client, 2026-10-26. Parse base_url with urlli"
                            "b.parse.urlsplit.\nPreserve unrelated query pairs in their "
                            "original order, including duplicate\nnames and blank values"
                            "; remove every exact lowercase limit, cursor and\narchived "
                            "pair. Append controlled pairs in order: limit, cursor, arc"
                            "hived.\nEncode all pairs with urllib.parse.urlencode, rebui"
                            "ld with urlunsplit and\nremove any URL fragment. Query key "
                            "matching is case sensitive.\n"
                        ),
                        task_key=("catalog-client"),
                    ),
                    Document(
                        path=("catalog/adapter.md"),
                        text=(
                            "Task catalog-client, API v1, 2026-10-26. A cursor is None "
                            "or a string. Omit\nNone and empty string; retain any other "
                            "cursor exactly, including whitespace.\noptions has optional"
                            " page_size integer, include_archived boolean and token\nstr"
                            "ing-or-None. Append archived=true only when include_archiv"
                            "ed is True.\nStrip token; if the result is nonempty set Aut"
                            "horization to 'Bearer ' plus\nit; otherwise headers is empt"
                            "y. No other headers. Do not mutate options.\n"
                        ),
                        task_key=("catalog-client"),
                    ),
                    Document(
                        path=("catalog/sidebar.md"),
                        text=(
                            "Task catalog-client, UI review, 2026-10-26. The sidebar sh"
                            "ows at most 25\nrows at a time and displays 'All' when arch"
                            "ived items are visible. That\nrendering cap is separate fro"
                            "m the transport page_size. Keep the request\nresult a dicti"
                            "onary so the caller can choose how to execute it.\n"
                        ),
                        task_key=("catalog-client"),
                    ),
                    Document(
                        path=("team/working-agreements.md"),
                        text=(
                            "Shared team note, 2026-10-28. Task labels identify separat"
                            "e maintenance streams.\nA note for another task describes t"
                            "hat team's caller; similar function names do\nnot change th"
                            "is repository's contract. Dated revisions apply to the nam"
                            "ed task.\nPatch solution.py using Python's standard library"
                            ". Our CI runs entirely offline.\n"
                        ),
                        task_key=(""),
                    ),
                    Document(
                        path=("search/transport.md"),
                        text=(
                            "Task search-panel, 2026-10-26. Our search endpoint uses si"
                            "ze=25, treats an\nempty cursor as an explicit checkpoint an"
                            "d retains URL fragments for client\nnavigation. These conve"
                            "ntions belong to the search panel, not the catalog.\n"
                        ),
                        task_key=("search-panel"),
                    ),
                ),
                test_source=(
                    "import copy\nimport unittest\nfrom solution import page_requ"
                    "est\n\nclass RequestChecks(unittest.TestCase):\n    def test_"
                    "query_policy(self):\n        base = 'https://example.test/i"
                    "tems'\n        self.assertEqual(page_request(base, {}),\n   "
                    "                      {'url': base + '?limit=50', 'headers"
                    "': {}})\n        self.assertEqual(page_request(base, {'page"
                    "_size': 0})['url'], base + '?limit=1')\n        self.assert"
                    "Equal(page_request(base, {'page_size': -5})['url'], base +"
                    " '?limit=1')\n        self.assertEqual(page_request(base, {"
                    "'page_size': 101})['url'], base + '?limit=100')\n        se"
                    "lf.assertEqual(page_request(base, {'page_size': 100})['url"
                    "'], base + '?limit=100')\n        self.assertEqual(page_req"
                    "uest(base, {}, '')['url'], base + '?limit=50')\n        sel"
                    "f.assertEqual(page_request(base, {}, ' a/+? ')['url'],\n   "
                    "                      base + '?limit=50&cursor=+a%2F%2B%3F"
                    "+')\n        self.assertEqual(page_request(base + '?x=1&lim"
                    "it=9&x=&cursor=old&archived=false#tab',\n                  "
                    "                    {'include_archived': True}, 'next')['u"
                    "rl'],\n                         base + '?x=1&x=&limit=50&cu"
                    "rsor=next&archived=true')\n        self.assertEqual(page_re"
                    "quest(base + '?Limit=7&limit=9&limit=10', {})['url'],\n    "
                    "                     base + '?Limit=7&limit=50')\n    def t"
                    "est_headers_and_purity(self):\n        options = {'token': "
                    "' abc ', 'page_size': 3, 'include_archived': False}\n      "
                    "  original = copy.deepcopy(options)\n        self.assertEqu"
                    "al(page_request('https://example.test/i#f', options),\n    "
                    "                     {'url': 'https://example.test/i?limit"
                    "=3',\n                          'headers': {'Authorization'"
                    ": 'Bearer abc'}})\n        self.assertEqual(options, origin"
                    "al)\n        self.assertEqual(page_request('https://example"
                    ".test/i', {'token': '  '})['headers'], {})\n        self.as"
                    "sertEqual(page_request('https://example.test/i', {'token':"
                    " None})['headers'], {})\n"
                ),
                reference_solution=(
                    "from urllib.parse import parse_qsl, urlencode, urlsplit, u"
                    "rlunsplit\n\ndef page_request(base_url, options, cursor=None"
                    '):\n    """Build the catalog GET request dictionary without'
                    ' mutating options."""\n    parts = urlsplit(base_url)\n    p'
                    "airs = [(key, value) for key, value in parse_qsl(parts.que"
                    "ry, keep_blank_values=True)\n             if key not in ('l"
                    "imit', 'cursor', 'archived')]\n    pairs.append(('limit', s"
                    "tr(max(1, min(100, options.get('page_size', 50))))))\n    i"
                    "f cursor:\n        pairs.append(('cursor', cursor))\n    if "
                    "options.get('include_archived', False):\n        pairs.appe"
                    "nd(('archived', 'true'))\n    token = (options.get('token')"
                    " or '').strip()\n    headers = {'Authorization': 'Bearer ' "
                    "+ token} if token else {}\n    url = urlunsplit((parts.sche"
                    "me, parts.netloc, parts.path, urlencode(pairs), ''))\n    r"
                    "eturn {'url': url, 'headers': headers}\n"
                ),
            ),
            Stage(
                id=("revision-2"),
                brief=(
                    "Revise solution.py for the October 28 catalog-client polic"
                    "y revision 2. Handle\nthe changed zero-size and empty-strin"
                    "g cursor boundaries from the updated\nsources. Keep API v1,"
                    " query replacement/order and authorization behavior.\nSearc"
                    "h and media clients are concurrent peer tasks with differe"
                    "nt conventions.\n"
                ),
                documents=(
                    Document(
                        path=("catalog/client.md"),
                        text=(
                            "Task catalog-client, request policy revision 2, 2026-10-28"
                            ". Replaces\nrevision 1 of October 26; API signature/return "
                            "shape remain v1. Missing\npage_size still means 50. Explici"
                            "t page_size=0 now requests the server's\ndefault, so omit l"
                            "imit entirely after removing old controlled parameters.\nOt"
                            "her integers still clamp to 1..100. page_request only cons"
                            "tructs a request.\n"
                        ),
                        task_key=("catalog-client"),
                    ),
                    Document(
                        path=("catalog/transport.md"),
                        text=(
                            "Task catalog-client, 2026-10-28. Preserve unrelated query "
                            "pairs, their order,\nduplicates and blank values using pars"
                            "e_qsl(keep_blank_values=True). Remove\nevery exact lowercas"
                            "e limit/cursor/archived key, then append whichever new\ncon"
                            "trolled pairs apply in limit,cursor,archived order. urlenc"
                            "ode all pairs;\nrebuild via urlsplit/urlunsplit with no fra"
                            "gment. Key matching stays case\nsensitive. An empty resulti"
                            "ng query must not leave a trailing question mark.\n"
                        ),
                        task_key=("catalog-client"),
                    ),
                    Document(
                        path=("catalog/adapter.md"),
                        text=(
                            "Task catalog-client, API v1, 2026-10-28, revised cursor bo"
                            "undary. cursor=None\nmeans absent; every string, including "
                            "'', is now an explicit checkpoint and\nmust be included as "
                            "a cursor pair. Preserve its exact value. include_archived\n"
                            "still adds archived=true only for True; default False. Tok"
                            "en is string/None;\ntrim and emit Authorization: Bearer <to"
                            "ken> only if nonblank. Return exactly\nurl and headers, nev"
                            "er change options; page_size integer types are valid.\n"
                        ),
                        task_key=("catalog-client"),
                    ),
                    Document(
                        path=("catalog/sidebar.md"),
                        text=(
                            "Task catalog-client, UI review, 2026-10-26. The sidebar sh"
                            "ows at most 25\nrows at a time and displays 'All' when arch"
                            "ived items are visible. That\nrendering cap is separate fro"
                            "m the transport page_size. Keep the request\nresult a dicti"
                            "onary so the caller can choose how to execute it.\n"
                        ),
                        task_key=("catalog-client"),
                    ),
                    Document(
                        path=("team/working-agreements.md"),
                        text=(
                            "Shared team note, 2026-10-28. Task labels identify separat"
                            "e maintenance streams.\nA note for another task describes t"
                            "hat team's caller; similar function names do\nnot change th"
                            "is repository's contract. Dated revisions apply to the nam"
                            "ed task.\nPatch solution.py using Python's standard library"
                            ". Our CI runs entirely offline.\n"
                        ),
                        task_key=(""),
                    ),
                    Document(
                        path=("search/transport.md"),
                        text=(
                            "Task search-panel, 2026-10-28. Search now clamps sizes thr"
                            "ough 200, omits\nempty cursors and adds X-Search-Version to"
                            " every request. This concurrent\nchange is for the search e"
                            "ndpoint's client and separate task label.\n"
                        ),
                        task_key=("search-panel"),
                    ),
                    Document(
                        path=("media/client.md"),
                        text=(
                            "Task media-index, 2026-10-28. Our image listing client tre"
                            "ats page_size=0\nas an empty page with limit=0 and preserve"
                            "s fragments for the gallery view.\nIts empty cursor restart"
                            "s a scan, so that client leaves the key out.\n"
                        ),
                        task_key=("media-index"),
                    ),
                ),
                test_source=(
                    "import copy\nimport unittest\nfrom solution import page_requ"
                    "est\n\nclass RequestChecks(unittest.TestCase):\n    def test_"
                    "current_query_policy(self):\n        base = 'https://exampl"
                    "e.test/items'\n        self.assertEqual(page_request(base, "
                    "{}),\n                         {'url': base + '?limit=50', "
                    "'headers': {}})\n        self.assertEqual(page_request(base"
                    ", {'page_size': 0})['url'], base)\n        self.assertEqual"
                    "(page_request(base + '?limit=9&cursor=old#f', {'page_size'"
                    ": 0})['url'],\n                         base)\n        self."
                    "assertEqual(page_request(base, {'page_size': -5})['url'], "
                    "base + '?limit=1')\n        self.assertEqual(page_request(b"
                    "ase, {'page_size': 101})['url'], base + '?limit=100')\n    "
                    "    self.assertEqual(page_request(base, {'page_size': 100}"
                    ")['url'], base + '?limit=100')\n        self.assertEqual(pa"
                    "ge_request(base, {}, '')['url'], base + '?limit=50&cursor="
                    "')\n        self.assertEqual(page_request(base, {'page_size"
                    "': 0}, '')['url'], base + '?cursor=')\n        self.assertE"
                    "qual(page_request(base, {}, ' a/+? ')['url'],\n            "
                    "             base + '?limit=50&cursor=+a%2F%2B%3F+')\n     "
                    "   self.assertEqual(page_request(base + '?x=1&limit=9&x=&c"
                    "ursor=old&archived=false#tab',\n                           "
                    "           {'include_archived': True}, 'next')['url'],\n   "
                    "                      base + '?x=1&x=&limit=50&cursor=next"
                    "&archived=true')\n        self.assertEqual(page_request(bas"
                    "e + '?Limit=7&limit=9&limit=10', {})['url'],\n             "
                    "            base + '?Limit=7&limit=50')\n    def test_heade"
                    "rs_and_purity(self):\n        options = {'token': ' abc ', "
                    "'page_size': 3, 'include_archived': False}\n        origina"
                    "l = copy.deepcopy(options)\n        self.assertEqual(page_r"
                    "equest('https://example.test/i#f', options),\n             "
                    "            {'url': 'https://example.test/i?limit=3',\n    "
                    "                      'headers': {'Authorization': 'Bearer"
                    " abc'}})\n        self.assertEqual(options, original)\n     "
                    "   self.assertEqual(page_request('https://example.test/i',"
                    " {'token': '  '})['headers'], {})\n        self.assertEqual"
                    "(page_request('https://example.test/i', {'token': None})['"
                    "headers'], {})\n"
                ),
                reference_solution=(
                    "from urllib.parse import parse_qsl, urlencode, urlsplit, u"
                    "rlunsplit\n\ndef page_request(base_url, options, cursor=None"
                    '):\n    """Build the catalog GET request dictionary without'
                    ' mutating options."""\n    parts = urlsplit(base_url)\n    p'
                    "airs = [(key, value) for key, value in parse_qsl(parts.que"
                    "ry, keep_blank_values=True)\n             if key not in ('l"
                    "imit', 'cursor', 'archived')]\n    size = options.get('page"
                    "_size', 50)\n    if size != 0:\n        pairs.append(('limit"
                    "', str(max(1, min(100, size)))))\n    if cursor is not None"
                    ":\n        pairs.append(('cursor', cursor))\n    if options."
                    "get('include_archived', False):\n        pairs.append(('arc"
                    "hived', 'true'))\n    token = (options.get('token') or '')."
                    "strip()\n    headers = {'Authorization': 'Bearer ' + token}"
                    " if token else {}\n    url = urlunsplit((parts.scheme, part"
                    "s.netloc, parts.path, urlencode(pairs), ''))\n    return {'"
                    "url': url, 'headers': headers}\n"
                ),
            ),
            Stage(
                id=("revision-3"),
                brief=(
                    "Maintain catalog-client under October 30 revision 3. The r"
                    "eplacement sources change the page-size ceiling and distin"
                    "guish missing include_archived from explicit False. Preser"
                    "ve the public API and the other documented request semanti"
                    "cs.\n"
                ),
                documents=(
                    Document(
                        path=("catalog/client.md"),
                        text=(
                            "Task catalog-client, request policy revision 3, 2026-10-30"
                            ". Replaces\nrevision 2 from October 28. Missing page_size s"
                            "till means 50; explicit zero\nomits limit, and negative int"
                            "egers clamp to 1. Positive integers now clamp\nthrough 200 "
                            "inclusive, replacing the old 100 cap. API v1 signature rem"
                            "ains\npage_request(base_url, options, cursor=None), returni"
                            "ng exactly url and headers.\n"
                        ),
                        task_key=("catalog-client"),
                    ),
                    Document(
                        path=("catalog/transport.md"),
                        text=(
                            "Task catalog-client, 2026-10-30. Still parse with urlsplit"
                            " and parse_qsl with\nkeep_blank_values=True; preserve unrel"
                            "ated query pair order, duplicates and\nblanks. Remove every"
                            " exact lowercase limit/cursor/archived pair. Append new\nap"
                            "plicable pairs in limit,cursor,archived order, then urlenc"
                            "ode and urlunsplit\nwith fragment removed. Keys are case se"
                            "nsitive. An empty query has no trailing ?.\nA missing inclu"
                            "de_archived option now omits archived; explicit True emits"
                            "\narchived=true and explicit False emits archived=false, re"
                            "placing false omission.\n"
                        ),
                        task_key=("catalog-client"),
                    ),
                    Document(
                        path=("catalog/adapter.md"),
                        text=(
                            "Task catalog-client, API v1, 2026-10-30. cursor=None means"
                            " absent; any string\nincluding empty or whitespace is an ex"
                            "plicit value and must be preserved.\noptions may contain in"
                            "teger page_size, boolean include_archived, string/None\ntok"
                            "en. Strip token; set only Authorization: Bearer <trimmed t"
                            "oken> when\nnonblank, otherwise empty headers. Do not mutat"
                            "e options or send any request.\nThis is a pure request-dict"
                            "ionary builder; no additional headers are needed.\n"
                        ),
                        task_key=("catalog-client"),
                    ),
                    Document(
                        path=("catalog/sidebar.md"),
                        text=(
                            "Task catalog-client, UI review, 2026-10-26. The sidebar sh"
                            "ows at most 25\nrows at a time and displays 'All' when arch"
                            "ived items are visible. That\nrendering cap is separate fro"
                            "m the transport page_size. Keep the request\nresult a dicti"
                            "onary so the caller can choose how to execute it.\n"
                        ),
                        task_key=("catalog-client"),
                    ),
                    Document(
                        path=("team/working-agreements.md"),
                        text=(
                            "Shared team note, 2026-10-28. Task labels identify separat"
                            "e maintenance streams.\nA note for another task describes t"
                            "hat team's caller; similar function names do\nnot change th"
                            "is repository's contract. Dated revisions apply to the nam"
                            "ed task.\nPatch solution.py using Python's standard library"
                            ". Our CI runs entirely offline.\n"
                        ),
                        task_key=(""),
                    ),
                    Document(
                        path=("search/transport.md"),
                        text=(
                            "Task search-panel, 2026-10-30. Search has returned to its "
                            "100-row cap and\nalways omits archived=false. Its empty cur"
                            "sor is omitted too. Keep these\nsearch conventions stable d"
                            "espite the catalog team's concurrent API work.\n"
                        ),
                        task_key=("search-panel"),
                    ),
                    Document(
                        path=("media/client.md"),
                        text=(
                            "Task media-index, 2026-10-30. Image requests use a 500-row"
                            " maximum and always\nadd Accept: image/webp. Our gallery ke"
                            "eps URL fragments because its output is\nalso a navigation "
                            "link. Media request dictionaries use separate options.\n"
                        ),
                        task_key=("media-index"),
                    ),
                ),
                test_source=(
                    "import copy\nimport unittest\nfrom solution import page_requ"
                    "est\n\nclass RequestChecks(unittest.TestCase):\n    def test_"
                    "current_query_boundaries(self):\n        base = 'https://ex"
                    "ample.test/items'\n        self.assertEqual(page_request(ba"
                    "se, {}),\n                         {'url': base + '?limit=5"
                    "0', 'headers': {}})\n        self.assertEqual(page_request("
                    "base, {'page_size': 0})['url'], base)\n        self.assertE"
                    "qual(page_request(base + '?limit=9&cursor=old#f', {'page_s"
                    "ize': 0})['url'],\n                         base)\n        s"
                    "elf.assertEqual(page_request(base, {'page_size': -5})['url"
                    "'], base + '?limit=1')\n        self.assertEqual(page_reque"
                    "st(base, {'page_size': 101})['url'], base + '?limit=101')\n"
                    "        self.assertEqual(page_request(base, {'page_size': "
                    "200})['url'], base + '?limit=200')\n        self.assertEqua"
                    "l(page_request(base, {'page_size': 201})['url'], base + '?"
                    "limit=200')\n        self.assertEqual(page_request(base, {}"
                    ", '')['url'], base + '?limit=50&cursor=')\n        self.ass"
                    "ertEqual(page_request(base, {'page_size': 0}, '')['url'], "
                    "base + '?cursor=')\n        self.assertEqual(page_request(b"
                    "ase, {}, ' a/+? ')['url'],\n                         base +"
                    " '?limit=50&cursor=+a%2F%2B%3F+')\n        self.assertEqual"
                    "(page_request(base + '?x=1&x=&archived=true#tab',\n        "
                    "                              {'include_archived': False},"
                    " 'next')['url'],\n                         base + '?x=1&x=&"
                    "limit=50&cursor=next&archived=false')\n        self.assertE"
                    "qual(page_request(base + '?Limit=7&limit=9',\n             "
                    "                         {'include_archived': True})['url'"
                    "],\n                         base + '?Limit=7&limit=50&arch"
                    "ived=true')\n    def test_headers_and_purity(self):\n       "
                    " options = {'token': ' abc ', 'page_size': 3, 'include_arc"
                    "hived': False}\n        original = copy.deepcopy(options)\n "
                    "       self.assertEqual(page_request('https://example.test"
                    "/i#f', options),\n                         {'url': 'https:/"
                    "/example.test/i?limit=3&archived=false',\n                 "
                    "         'headers': {'Authorization': 'Bearer abc'}})\n    "
                    "    self.assertEqual(options, original)\n        self.asser"
                    "tEqual(page_request('https://example.test/i', {'token': ' "
                    " '})['headers'], {})\n        self.assertEqual(page_request"
                    "('https://example.test/i', {'token': None})['headers'], {}"
                    ")\n"
                ),
                reference_solution=(
                    "from urllib.parse import parse_qsl, urlencode, urlsplit, u"
                    "rlunsplit\n\ndef page_request(base_url, options, cursor=None"
                    '):\n    """Build the catalog GET request dictionary without'
                    ' mutating options."""\n    parts = urlsplit(base_url)\n    p'
                    "airs = [(key, value) for key, value in parse_qsl(parts.que"
                    "ry, keep_blank_values=True)\n             if key not in ('l"
                    "imit', 'cursor', 'archived')]\n    size = options.get('page"
                    "_size', 50)\n    if size != 0:\n        pairs.append(('limit"
                    "', str(max(1, min(200, size)))))\n    if cursor is not None"
                    ":\n        pairs.append(('cursor', cursor))\n    if 'include"
                    "_archived' in options:\n        pairs.append(('archived', '"
                    "true' if options['include_archived'] else 'false'))\n    to"
                    "ken = (options.get('token') or '').strip()\n    headers = {"
                    "'Authorization': 'Bearer ' + token} if token else {}\n    u"
                    "rl = urlunsplit((parts.scheme, parts.netloc, parts.path, u"
                    "rlencode(pairs), ''))\n    return {'url': url, 'headers': h"
                    "eaders}\n"
                ),
            ),
        ),
    ),
)


def manifest_json() -> str:
    """Return the canonical sealed fixture manifest, including authored checks."""
    return json.dumps(
        {"version": 1, "cases": [asdict(case) for case in CASES]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def manifest_sha256() -> str:
    return hashlib.sha256(manifest_json().encode("utf-8")).hexdigest()
