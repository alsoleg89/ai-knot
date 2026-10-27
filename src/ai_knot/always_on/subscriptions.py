"""Owner-scoped durable change delivery with explicit acknowledgements."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from ai_knot.always_on.changes import ChangeBatch, FactChange, canonical, visible
from ai_knot.always_on.models import AccessDenied, CursorExpired, InvalidDelivery, RevisionConflict
from ai_knot.always_on.storage import (
    AlwaysOnSession,
    AlwaysOnStorageCapable,
    SQLRow,
    SQLStatement,
    enable_stream,
    sql,
)

if TYPE_CHECKING:
    from ai_knot.pool import SharedMemoryPool


@dataclass(frozen=True)
class WatchSelector:
    namespace: str
    fact_ids: tuple[str, ...] = ()
    slot_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class Subscription:
    id: str
    owner_agent_id: str
    selector: WatchSelector
    cursor: int
    debounce_seconds: float


def page_limit(limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 500:
        raise ValueError("limit must be an integer from 1 to 500")


def key(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nonempty identifier required")


class SubscriptionsMixin:
    _storage: AlwaysOnStorageCapable
    _owner: str
    _pool: SharedMemoryPool | None
    _clock: Callable[[], datetime]

    def _now(self) -> datetime:
        raise NotImplementedError

    def _transaction(
        self, namespaces: tuple[str, ...] = (), *, write: bool = True
    ) -> AbstractContextManager[AlwaysOnSession]:
        raise NotImplementedError

    def _subscription_namespace(self, namespace: str) -> None:
        if namespace == self._owner:
            return
        pool = self._pool
        if namespace != "__shared__" or pool is None or not pool._persist_stats:
            raise AccessDenied("namespace unavailable")
        a: object = self._storage
        b: object = pool._storage
        if a is b:
            return
        if getattr(a, "_db_path", None) and getattr(a, "_db_path", None) == getattr(
            b, "_db_path", None
        ):
            return
        if getattr(a, "_dsn", None) and getattr(a, "_dsn", None) == getattr(b, "_dsn", None):
            return
        raise AccessDenied("pool must use the same durable store")

    def _subrow(self, session: AlwaysOnSession, subscription_id: str) -> SQLRow:
        row = session.fetch_one(
            sql("SELECT * FROM ao_subscriptions WHERE owner=? AND id=?"),
            (self._owner, subscription_id),
        )
        if row is None:
            raise AccessDenied("subscription unavailable")
        return row

    @staticmethod
    def _subscription(row: SQLRow) -> Subscription:
        selector = json.loads(str(row["selector_json"]))
        return Subscription(
            str(row["id"]),
            str(row["owner"]),
            WatchSelector(
                selector["namespace"], tuple(selector["fact_ids"]), tuple(selector["slot_keys"])
            ),
            int(row["cursor"] or 0),
            float(row["debounce_seconds"] or 0),
        )

    def subscribe(
        self,
        *,
        subscription_id: str,
        selector: WatchSelector,
        start: Literal["now", "beginning"] = "now",
        debounce_seconds: float = 0,
    ) -> Subscription:
        key(subscription_id)
        self._subscription_namespace(selector.namespace)
        if start not in ("now", "beginning"):
            raise ValueError("start must be now or beginning")
        if (
            isinstance(debounce_seconds, bool)
            or not isinstance(debounce_seconds, (int, float))
            or not math.isfinite(debounce_seconds)
            or not 0 <= debounce_seconds <= 3600
        ):
            raise ValueError("debounce_seconds must be finite from 0 to 3600")
        for value in (*selector.fact_ids, *selector.slot_keys):
            key(value)
        selector = replace(
            selector,
            fact_ids=tuple(sorted(set(selector.fact_ids))),
            slot_keys=tuple(sorted(set(selector.slot_keys))),
        )
        encoded = canonical(asdict(selector))
        if len(encoded.encode()) > 65536:
            raise ValueError("selector exceeds 65536 bytes")
        now = self._now()
        with self._transaction((selector.namespace,)) as session:
            enable_stream(session, selector.namespace, now)
            existing = session.fetch_one(
                sql("SELECT * FROM ao_subscriptions WHERE owner=? AND id=?"),
                (self._owner, subscription_id),
            )
            if existing:
                if (
                    existing["selector_json"] != encoded
                    or existing["debounce_seconds"] != debounce_seconds
                ):
                    raise RevisionConflict("subscription contract differs")
                return self._subscription(existing)
            stream = session.fetch_one(
                sql("SELECT * FROM ao_streams WHERE namespace=?"), (selector.namespace,)
            )
            assert stream is not None
            cursor = int(stream["head_seq"] or 0) if start == "now" else 0
            if cursor < int(stream["min_retained_seq"] or 1) - 1:
                raise CursorExpired("requested history was pruned")
            session.execute(
                sql(
                    "INSERT INTO ao_subscriptions(owner,id,incarnation,namespace,selector_j"
                    "son,cursor,debounce_seconds) VALUES(?,?,?,?,?,?,?)"
                ),
                (
                    self._owner,
                    subscription_id,
                    uuid.uuid4().hex,
                    selector.namespace,
                    encoded,
                    cursor,
                    debounce_seconds,
                ),
            )
            return self._subscription(self._subrow(session, subscription_id))

    def _grants(self, session: AlwaysOnSession, namespace: str) -> set[str]:
        if namespace != "__shared__":
            return set()
        return {
            str(row["scope"])
            for row in session.fetch_all(
                sql("SELECT scope FROM acl_grants WHERE agent_id=?"), (self._owner,)
            )
        }

    def _current_visible(
        self,
        session: AlwaysOnSession,
        namespace: str,
        fact_id: str,
        projections: dict[str, str],
        grants: set[str],
    ) -> bool:
        if namespace == self._owner:
            return True
        projection = projections.get(fact_id)
        if projection is None:
            current = session.fetch_one(
                SQLStatement(
                    "SELECT origin_agent_id,visibility_scope,visibility,topic_channel,memory_tier "
                    "FROM facts WHERE agent_id=? AND id=?",
                    "SELECT origin_agent_id,visibility_scope,visibility,topic_channel,memory_tier "
                    'FROM "ai-knot_facts" WHERE agent_id=%s AND id=%s',
                ),
                (namespace, fact_id),
            )
            if current is not None:
                projection = canonical(current)
            else:
                latest = session.fetch_one(
                    sql(
                        "SELECT old_access_json,new_access_json FROM ao_changes "
                        "WHERE namespace=? AND fact_id=? ORDER BY seq DESC LIMIT 1"
                    ),
                    (namespace, fact_id),
                )
                if latest is None:
                    return False
                projection = str(latest["new_access_json"] or latest["old_access_json"])
            projections[fact_id] = projection
        return visible(projection, self._owner, grants)

    def poll(self, subscription_id: str, *, limit: int = 100) -> ChangeBatch:
        page_limit(limit)
        key(subscription_id)
        # Discover the namespace, then lock it together with owner before re-read.
        with self._transaction(write=False) as session:
            namespace = str(self._subrow(session, subscription_id)["namespace"])
        self._subscription_namespace(namespace)
        with self._transaction((namespace,)) as session:
            row = self._subrow(session, subscription_id)
            if row["namespace"] != namespace:
                raise RevisionConflict("subscription changed")
            stream = session.fetch_one(
                sql("SELECT * FROM ao_streams WHERE namespace=?"), (namespace,)
            )
            assert stream is not None
            if int(row["cursor"] or 0) < int(stream["min_retained_seq"] or 1) - 1:
                raise CursorExpired("subscription history was pruned")
            grants = self._grants(session, namespace)
            acl_digest = hashlib.sha256(canonical(sorted(grants)).encode()).hexdigest()
            projections: dict[str, str] = {}
            if row["inflight_token"] is not None:
                events = tuple(
                    FactChange(**event) for event in json.loads(str(row["inflight_events_json"]))
                )
                resync = bool(row["inflight_resync"])
                if row["inflight_acl_digest"] != acl_digest or any(
                    not self._current_visible(
                        session, namespace, event.fact_id, projections, grants
                    )
                    for event in events
                ):
                    events, resync = (), True
                    session.execute(
                        sql(
                            "UPDATE ao_subscriptions SET inflight_events_json=?,inflight_resync=1,i"
                            "nflight_acl_digest=? WHERE owner=? AND id=?"
                        ),
                        ("[]", acl_digest, self._owner, subscription_id),
                    )
                return ChangeBatch(str(row["inflight_token"]), events, resync)
            rows = session.fetch_all(
                sql("SELECT * FROM ao_changes WHERE namespace=? AND seq>? ORDER BY seq LIMIT 1000"),
                (namespace, row["cursor"]),
            )
            if not rows:
                return ChangeBatch(None, ())
            selector = self._subscription(row).selector
            events_by_id: dict[str, FactChange] = {}
            resync = False
            through = int(row["cursor"] or 0)
            for event in rows:
                fact_id = str(event["fact_id"])
                if event["kind"] == "acl_changed":
                    resync |= event["acl_subject"] == self._owner
                elif event["kind"] == "resync_required":
                    resync = True
                elif (
                    not selector.fact_ids
                    and not selector.slot_keys
                    or fact_id in selector.fact_ids
                    or event["slot_key"] in selector.slot_keys
                    or event["before_slot_key"] in selector.slot_keys
                ):
                    old_access = str(event["old_access_json"]) if event["old_access_json"] else None
                    new_access = str(event["new_access_json"]) if event["new_access_json"] else None
                    allowed = namespace == self._owner or visible(
                        new_access or old_access, self._owner, grants
                    )
                    previously = namespace == self._owner or visible(
                        old_access, self._owner, grants
                    )
                    current = self._current_visible(
                        session, namespace, fact_id, projections, grants
                    )
                    if (allowed or previously) and not current:
                        resync = True
                    elif allowed and current:
                        if fact_id not in events_by_id and len(events_by_id) == limit:
                            break
                        previous = events_by_id.get(fact_id)
                        events_by_id[fact_id] = FactChange(
                            namespace,
                            int(event["seq"] or 0),
                            str(event["kind"]),
                            fact_id,
                            str(event["slot_key"]),
                            previous.before_revision
                            if previous
                            else (
                                str(event["before_revision"]) if event["before_revision"] else None
                            ),
                            str(event["after_revision"]) if event["after_revision"] else None,
                        )
                through = int(event["seq"] or 0)
            if (events_by_id or resync) and float(row["debounce_seconds"] or 0) > 0:
                now = self._now()
                pending = (
                    datetime.fromisoformat(str(row["pending_since"]))
                    if row["pending_since"]
                    else now
                )
                if (now - pending).total_seconds() < float(row["debounce_seconds"] or 0):
                    if row["pending_since"] is None:
                        session.execute(
                            sql(
                                "UPDATE ao_subscriptions SET pending_since=? WHERE owner=? AND id=?"
                            ),
                            (now.isoformat(timespec="microseconds"), self._owner, subscription_id),
                        )
                    return ChangeBatch(None, ())
            events = tuple(events_by_id.values())
            token = uuid.uuid4().hex
            session.execute(
                sql(
                    "UPDATE ao_subscriptions SET pending_since=NULL,inflight_token=?,inflig"
                    "ht_through_seq=?,inflight_events_json=?,inflight_resync=?,inflight_acl"
                    "_digest=? WHERE owner=? AND id=?"
                ),
                (
                    token,
                    through,
                    canonical([asdict(event) for event in events]),
                    int(resync),
                    acl_digest,
                    self._owner,
                    subscription_id,
                ),
            )
            return ChangeBatch(token, events, resync)

    def ack(self, subscription_id: str, *, delivery_id: str) -> None:
        key(subscription_id)
        key(delivery_id)
        with self._transaction() as session:
            row = self._subrow(session, subscription_id)
            receipt = session.fetch_one(
                sql(
                    "SELECT through_seq FROM ao_delivery_receipts WHERE owner=? AND subscri"
                    "ption_id=? AND incarnation=? AND delivery_id=?"
                ),
                (self._owner, subscription_id, row["incarnation"], delivery_id),
            )
            if receipt is not None:
                return
            if row["inflight_token"] != delivery_id:
                raise InvalidDelivery("unknown delivery")
            session.execute(
                sql(
                    "INSERT INTO ao_delivery_receipts(owner,subscription_id,incarnation,del"
                    "ivery_id,through_seq) VALUES(?,?,?,?,?)"
                ),
                (
                    self._owner,
                    subscription_id,
                    row["incarnation"],
                    delivery_id,
                    row["inflight_through_seq"],
                ),
            )
            session.execute(
                sql(
                    "UPDATE ao_subscriptions SET cursor=inflight_through_seq,inflight_token"
                    "=NULL,inflight_through_seq=NULL,inflight_events_json=NULL,inflight_res"
                    "ync=NULL,inflight_acl_digest=NULL WHERE owner=? AND id=?"
                ),
                (self._owner, subscription_id),
            )

    def unsubscribe(self, subscription_id: str) -> None:
        key(subscription_id)
        with self._transaction() as session:
            session.execute(
                sql("DELETE FROM ao_delivery_receipts WHERE owner=? AND subscription_id=?"),
                (self._owner, subscription_id),
            )
            session.execute(
                sql("DELETE FROM ao_subscriptions WHERE owner=? AND id=?"),
                (self._owner, subscription_id),
            )

    def prune_changes(
        self, namespace: str | None = None, *, through_seq: int | None = None, force: bool = False
    ) -> int:
        namespace = namespace or self._owner
        self._subscription_namespace(namespace)
        if through_seq is not None and (type(through_seq) is not int or through_seq < 0):
            raise ValueError("nonnegative through_seq required")
        with self._transaction((namespace,)) as session:
            stream = session.fetch_one(
                sql("SELECT head_seq FROM ao_streams WHERE namespace=?"), (namespace,)
            )
            if stream is None:
                return 0
            frontier = int(stream["head_seq"] or 0)
            if not force:
                row = session.fetch_one(
                    sql("SELECT MIN(cursor) AS frontier FROM ao_subscriptions WHERE namespace=?"),
                    (namespace,),
                )
                if row and row["frontier"] is not None:
                    frontier = min(frontier, int(row["frontier"]))
            if through_seq is not None:
                frontier = min(frontier, through_seq)
            count = session.execute(
                sql("DELETE FROM ao_changes WHERE namespace=? AND seq<=?"), (namespace, frontier)
            )
            session.execute(
                sql(
                    "UPDATE ao_streams SET min_retained_seq=? "
                    "WHERE namespace=? AND min_retained_seq<?"
                ),
                (frontier + 1, namespace, frontier + 1),
            )
            return count
