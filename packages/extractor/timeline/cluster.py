"""Group located entities into date-anchored events.

The clustering rule is simple, deterministic, and (importantly) easy
to explain on a UI tooltip:

1. Walk the located entities in source order (by ``start``).
2. The first ``date`` entity opens an event. Every non-date entity
   we then encounter, **as long as it falls within the configured
   character window of the most recent date**, is attached to that
   event.
3. The next ``date`` entity opens a new event.
4. Located entities that fall outside any window go to a single
   *ambient* bucket per doc — the case-level facts that have no
   date anchor.
5. Unlocated entities (``start is None``) all go to the ambient
   bucket too, regardless of type.

Tuning ``cluster_window_chars`` is a single knob that trades off
event resolution against orphaning rate:

* Too small → many "barely later" facts orphan to ambient.
* Too large → unrelated paragraphs collapse into one event.

The default ``1500`` characters is calibrated on the 140-doc sample:
roughly the length of one paragraph of a Vietnamese ban-án after the
preamble. Override per call from the config.

The output is a list of :class:`Cluster` records with the date entity
(if any) and the entities attached to it. The downstream classifier
(``classify.py``) consumes this and the source text to assign
:class:`packages.extractor.timeline.schema.EventKind` labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from packages.extractor.timeline.locator import LocatedEntity


@dataclass
class Cluster:
    """A group of entities anchored to (at most) one ``date`` entity.

    ``anchor`` is the single ``date`` entity that opens the event;
    ``members`` are every other entity attached to it (parties,
    locations, money, statutes, ...). For ambient clusters
    ``anchor is None`` and ``members`` carries every unanchored
    entity in the doc.
    """

    anchor: LocatedEntity | None
    members: list[LocatedEntity] = field(default_factory=list)

    @property
    def char_start(self) -> int | None:
        """Earliest ``start`` across the anchor + members, or None."""
        starts = [
            x.start for x in ([self.anchor] if self.anchor else []) + self.members
            if x.start is not None
        ]
        return min(starts) if starts else None

    @property
    def char_end(self) -> int | None:
        """Latest ``end`` across the anchor + members, or None."""
        ends = [
            x.end for x in ([self.anchor] if self.anchor else []) + self.members
            if x.end is not None
        ]
        return max(ends) if ends else None


def cluster_by_date_proximity(
    located: list[LocatedEntity],
    *,
    cluster_window_chars: int = 1500,
) -> tuple[list[Cluster], Cluster]:
    """Partition located entities into ``(dated_clusters, ambient)``.

    Both outputs contain :class:`LocatedEntity` instances. Dated
    clusters are returned in document order (by ``Cluster.char_start``).
    The ``ambient`` cluster carries every entity that did not fit a
    window — regardless of whether it was located or not — so callers
    have a single place to look for case-level facts.
    """
    if cluster_window_chars < 0:
        raise ValueError("cluster_window_chars must be non-negative")

    # Stable sort: located entities first (by start), unlocated last
    # (preserving input order via ``enumerate`` as a tiebreaker).
    indexed = list(enumerate(located))
    indexed.sort(key=lambda pair: (
        pair[1].start if pair[1].start is not None else float("inf"),
        pair[0],
    ))

    dated: list[Cluster] = []
    ambient = Cluster(anchor=None, members=[])
    open_cluster: Cluster | None = None
    open_anchor_end: int | None = None

    for _, le in indexed:
        # Unlocated entities always go to ambient.
        if le.start is None:
            ambient.members.append(le)
            continue

        # A new date opens a new cluster.
        if le.entity.type == "date":
            open_cluster = Cluster(anchor=le, members=[])
            open_anchor_end = le.end
            dated.append(open_cluster)
            continue

        # Non-date entity: attach to the open cluster if its start
        # is within the window of the last anchor's end.
        if (
            open_cluster is not None
            and open_anchor_end is not None
            and le.start - open_anchor_end <= cluster_window_chars
        ):
            open_cluster.members.append(le)
        else:
            ambient.members.append(le)

    return dated, ambient


__all__ = ["Cluster", "cluster_by_date_proximity"]
