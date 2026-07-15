"""Platform-specific adapters onto the canonical ArLeS schema (see arles.schema).

A mapper's whole job is to turn one platform's export into canonical actions. Keeping
that knowledge here is what lets the metrics stay platform-agnostic: `bluesky.py` is
the only module in the codebase that knows what an AT-URI is.

To support a new platform, write a sibling module exposing `map_row(row)` and
`convert(source, destination)`. The contract is arles.schema.CanonicalAction, and the
one requirement worth reading twice is `parent_actor_id`: without it the
super-spreader axis cannot be estimated, and no metric can recover it for you.
"""
