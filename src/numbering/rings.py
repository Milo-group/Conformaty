"""Ring-completion numbering stage (always SSSR on query/ref molecules)."""

from rdkit import Chem

from numbering.context import RenumberingState


def _sssr(mol):
    return [list(ring) for ring in Chem.GetSSSR(mol)]


def _rotate(seq, first):
    seq = list(seq)
    if first not in seq:
        return None
    i = seq.index(first)
    return seq[i:] + seq[:i]


def _in_ring_matches(q_ring, r_ring, mapping):
    """Query ring atoms mapped to an atom inside `r_ring`."""
    r_set = set(r_ring)
    return [q for q in q_ring if q in mapping and mapping[q] in r_set]


def _can_override(q, mapping, seed_proximity, protected):
    """True if this query assignment may be displaced by a ring walk.

    Allowed: first coords_proximity pairs (and other unlocked pairs, e.g. MCS seed).
    Forbidden: pairs locked by extend_mcs / complete_rings / coords_proximity_any.
    """
    if q in protected and protected.get(q) == mapping.get(q):
        return False
    if q in seed_proximity and seed_proximity.get(q) == mapping.get(q):
        return True
    # Unlocked non-proximity (typically init MCS) may still be displaced.
    return q not in protected


def _walk_propose(q_ring, r_ring, mapping, seed_proximity, protected):
    """Propose a full query-ring→ref-ring map by walking, or None if impossible.

    May drop soft pairs (first proximity / unlocked MCS), but not locked pairs
    from extend_mcs or complete_rings. Returns (proposed_ring_map, drop_set).
    """
    if len(q_ring) != len(r_ring):
        return None

    r_set = set(r_ring)
    mapped_q = _in_ring_matches(q_ring, r_ring, mapping)
    if len(mapped_q) < 3:
        return None

    proposals = {}
    for seed_q in mapped_q:
        q_ord = _rotate(q_ring, seed_q)
        r_ord = _rotate(r_ring, mapping[seed_q])
        if q_ord is None or r_ord is None:
            continue

        aligned = True
        for i, qi in enumerate(q_ord):
            if qi not in mapping or mapping[qi] not in r_set:
                continue
            if mapping[qi] != r_ord[i]:
                aligned = False
                break
        if not aligned:
            continue

        proposed = {qi: r_ord[i] for i, qi in enumerate(q_ord)}
        drop = set()
        ok = True

        for qi, ri in proposed.items():
            if qi in mapping and mapping[qi] != ri:
                if not _can_override(qi, mapping, seed_proximity, protected):
                    ok = False
                    break
                drop.add(qi)

            holders = [q for q, r in mapping.items() if r == ri and q != qi]
            for hq in holders:
                if not _can_override(hq, mapping, seed_proximity, protected):
                    ok = False
                    break
                drop.add(hq)
            if not ok:
                break

        if not ok:
            continue

        key = tuple(sorted(proposed.items()))
        proposals[key] = (proposed, drop)

    if len(proposals) != 1:
        return None
    return next(iter(proposals.values()))


def complete_from_mol_rings(mapping, mol_query, mol_ref, seed_proximity=None, protected=None):
    """Complete SSSR rings from ≥3 matched atoms (proximity or MCS).

    Ambiguity (skip numbering) only among options of the *same ring size*:
    - several distinct walks for one query ring, or
    - several same-size query rings whose walks share query/ref atoms.

    Different-size overlapping rings are not treated as alternatives: each
    unique size may be numbered.     Soft pairs may be dropped; locked pairs from
    extend_mcs / complete_rings are not.

    Returns (added_or_updated, dropped_query_idxs, confirmed_ring_pairs).
    `confirmed_ring_pairs` is the full walk map for every ring that was
    accepted (including already-complete rings with nothing new to add).
    """
    seed_proximity = dict(seed_proximity or {})
    protected = dict(protected or {})
    added = {}
    dropped = set()
    confirmed = {}
    occupied = dict(mapping)
    ref_rings = _sssr(mol_ref)

    # Collect unique walks per query ring first; apply only unambiguous ones.
    per_ring = []
    for q_ring in _sssr(mol_query):
        ring_hits = [q for q in q_ring if q in occupied]
        if len(ring_hits) < 3:
            continue

        candidates = {}
        for r_ring in ref_rings:
            if len(q_ring) != len(r_ring):
                continue
            if len(_in_ring_matches(q_ring, r_ring, occupied)) < 3:
                continue
            for r_ord in (list(r_ring), list(reversed(r_ring))):
                result = _walk_propose(
                    q_ring, r_ord, occupied, seed_proximity, protected,
                )
                if result is None:
                    continue
                proposed, drop = result
                key = tuple(sorted(proposed.items()))
                candidates[key] = (proposed, drop)

        # Several same-size fills for this ring → ambiguous, skip.
        if len(candidates) != 1:
            continue
        proposed, drop = next(iter(candidates.values()))
        per_ring.append((list(q_ring), proposed, drop))

    # Overlap groups: skip only when multiple rings of the *same size* compete.
    # Different sizes are not alternatives — keep each size's singleton.
    parent = list(range(len(per_ring)))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i, (_qr_i, prop_i, _drop_i) in enumerate(per_ring):
        refs_i = set(prop_i.values())
        qs_i = set(prop_i)
        for j, (_qr_j, prop_j, _drop_j) in enumerate(per_ring):
            if j <= i:
                continue
            if refs_i & set(prop_j.values()) or qs_i & set(prop_j):
                _union(i, j)

    components = {}
    for i in range(len(per_ring)):
        components.setdefault(_find(i), []).append(i)

    skip = set()
    for members in components.values():
        by_size = {}
        for i in members:
            by_size.setdefault(len(per_ring[i][0]), []).append(i)
        for _size, idxs in by_size.items():
            if len(idxs) > 1:
                skip.update(idxs)

    # Apply larger rings first so different-size overlaps are stable.
    order = sorted(
        (i for i in range(len(per_ring)) if i not in skip),
        key=lambda i: -len(per_ring[i][0]),
    )
    for i in order:
        _q_ring, proposed, drop = per_ring[i]
        for q in drop:
            occupied.pop(q, None)
            dropped.add(q)
            added.pop(q, None)
            confirmed.pop(q, None)
        for qi, ri in proposed.items():
            prev = occupied.get(qi)
            if prev != ri:
                added[qi] = ri
            occupied[qi] = ri
            confirmed[qi] = ri

    return added, dropped, confirmed


def _ring_seeds(state: RenumberingState) -> dict:
    """Matched pairs for ring walking: MCS/mapping plus proximity (proximity wins ties)."""
    seeds = {}
    if state.atoms_mcs_map:
        seeds.update(state.atoms_mcs_map)
    seeds.update(state.mapping)
    seeds.update(state.overlap_mapping)
    return seeds


def complete_rings(state: RenumberingState) -> RenumberingState:
    """Complete rings from query/ref SSSR by walking ≥3 matched atoms."""
    seeds = _ring_seeds(state)
    added, dropped, confirmed = complete_from_mol_rings(
        seeds,
        state.mol_query,
        state.mol_ref,
        seed_proximity=state.seed_proximity_mapping,
        protected=state.locked_mapping,
    )

    for q in dropped:
        state.mapping.pop(q, None)
        state.overlap_mapping.pop(q, None)
        state.locked_mapping.pop(q, None)

    # Commit + lock the full confirmed walk (even if already complete / +0 added)
    # so extend_mcs keeps those pairs via locked_mapping.
    state.overlap_mapping.update(confirmed)
    state.mapping.update(confirmed)
    state.locked_mapping.update(confirmed)

    note = f"+{len(added)} ring atoms; SSSR walk"
    if confirmed:
        note += f"; locked {len(confirmed)} ring pair(s)"
    if dropped:
        note += f"; dropped {len(dropped)} soft pair(s)"
    state.record("complete_rings", notes=note)
    return state
