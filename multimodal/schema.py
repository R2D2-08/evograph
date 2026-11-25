"""Frozen EMBFV v1 dimensions and graph metadata."""

SCHEMA_VERSION = "evograph-embfv-v1"
STATIC_DIM = 2381
STATS_DIM = 128
DYNAMIC_DIM = 2048
FEATURE_DIM = STATIC_DIM + DYNAMIC_DIM
BLOCK_DIMS = {
    "statistics": 128,
    "api_identity": 512,
    "api_sequences": 256,
    "files": 256,
    "processes": 256,
    "registry": 128,
    "network": 128,
    "other_resources": 128,
    "arguments": 128,
    "payloads": 128,
}
BLOCK_SLICES = {}
_offset = 0
for _name, _size in BLOCK_DIMS.items():
    BLOCK_SLICES[_name] = slice(_offset, _offset + _size)
    _offset += _size
assert _offset == DYNAMIC_DIM

NODE_TYPES = ("binary", "process", "api", "window", "resource", "payload")
FORWARD_RELATIONS = (
    ("binary", "launches", "process"),
    ("binary", "imports", "api"),
    ("process", "spawns", "process"),
    ("process", "calls", "api"),
    ("process", "executes", "window"),
    ("window", "contains_api", "api"),
    ("window", "next_window", "window"),
    ("window", "reads", "resource"),
    ("window", "writes", "resource"),
    ("window", "deletes", "resource"),
    ("window", "modifies", "resource"),
    ("window", "connects_to", "resource"),
    ("window", "accesses", "resource"),
    ("process", "loads", "resource"),
    ("process", "accesses", "resource"),
    ("process", "drops", "payload"),
    ("binary", "produces", "payload"),
)
RELATIONS = FORWARD_RELATIONS + tuple(
    (dst, "rev_" + relation, src) for src, relation, dst in FORWARD_RELATIONS
)
