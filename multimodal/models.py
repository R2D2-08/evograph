"""Trainable EMBFV baseline and six-type relational EvoGraph classifiers.

EvoGraphClassifier(mode='hybrid', hidden=256, dropout=.2, window_size=32,
                   max_windows=256)(samples) returns raw logits of shape [B].
Each sample contains parse_report IR in ``report``, ``static_features`` (2381
finite values or None), and pre-standardized ``statistics`` (128 finite values).
Payload IR records may also contain ``static_features``. Use BCEWithLogitsLoss;
apply sigmoid only for inference. No extraction, scaling, or binary execution
occurs here. All model modes work on CPU and follow Module.to(device).

Modes: baseline = static + all EMBFV blocks; static = only static; dynamic =
only EMBFV; graph = static binary node + behavioral graph, without D1; hybrid =
graph + D1. Graph/hybrid never compute the baseline DynamicEncoder. Graph caps
apply only to graph windows, not to full count features or baseline streams.
"""

import torch
from torch import nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import scatter, softmax

from .encoders import BehavioralEncoders, DynamicEncoder, feature_tensor, stable_bucket
from .graph import BINARY_TELEMETRY_NAMES, RESOURCE_STAT_NAMES, build_graph
from .schema import DYNAMIC_DIM, NODE_TYPES, RELATIONS, STATIC_DIM, STATS_DIM


class StaticEncoder(nn.Module):
    """Mask-aware 2381 -> 512 -> 256 encoder; None is not an all-zero PE."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(STATIC_DIM, 512), nn.GELU(), nn.Linear(512, 256))
        self.availability = nn.Embedding(2, 256)

    def forward(self, values, available):
        return self.mlp(values) * available[:, None] + self.availability(available.long())


class RelationalLayer(nn.Module):
    """R-GCN formula: sum_r mean_{u in N_r(v)} W_r h_u + W_type h_v.

    SAGEConv here is only a relation-specific linear mean aggregator, with
    root/bias/normalization disabled. HeteroConv sums those relation means,
    rather than degree-normalizing across unrelated edge types. A type-specific
    self transform, residual, GELU, dropout and LayerNorm complete the layer.
    """

    def __init__(self, hidden, dropout):
        super().__init__()
        self.relations = HeteroConv({relation: SAGEConv((hidden, hidden), hidden, aggr="mean",
                                                       root_weight=False, bias=False)
                                     for relation in RELATIONS}, aggr="sum")
        self.self_projection = nn.ModuleDict({kind: nn.Linear(hidden, hidden) for kind in NODE_TYPES})
        self.norm = nn.ModuleDict({kind: nn.LayerNorm(hidden) for kind in NODE_TYPES})
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        messages = self.relations(x, {key: edges for key, edges in edge_index.items() if edges.size(1)})
        return {kind: self.norm[kind](values + self.dropout(torch.nn.functional.gelu(
            self.self_projection[kind](values) + messages.get(kind, torch.zeros_like(values)))))
                for kind, values in x.items()}


class GraphEncoder(nn.Module):
    """Raw node encoding, tensor-only PyG batching, 3 layers, type attention.

    Process inputs are name64 + projected command32 + local statistics32;
    baseline D5 retains name64 + command64. Resource inputs combine shared
    string encoders, type embeddings and projected operation/evidence statistics.
    Binary inputs add a learned coverage/truncation projection to the 256-dim
    static embedding, so graph-only mode preserves missingness without D1.
    """

    def __init__(self, hidden=256, dropout=.2):
        super().__init__()
        self.static = StaticEncoder()
        self.binary_telemetry = nn.Linear(len(BINARY_TELEMETRY_NAMES), 256, bias=False)
        self.command_projection = nn.Linear(64, 32)
        self.api_projection = nn.Linear(74, 128)
        self.window_projection = nn.Linear(256, 128)
        self.file_projection = nn.Linear(128, 128)
        self.small_resource_projection = nn.ModuleDict({kind: nn.Linear(64, 128)
                                                        for kind in ("registry", "network", "other")})
        self.resource_types = nn.Embedding(7, 128)
        self.resource_statistics = nn.Linear(len(RESOURCE_STAT_NAMES), 128, bias=False)
        self.payload_projection = nn.Linear(64, 128)
        self.initial_projection = nn.ModuleDict({kind: nn.Linear(256 if kind == "binary" else 128, hidden)
                                                 for kind in NODE_TYPES})
        self.layers = nn.ModuleList(RelationalLayer(hidden, dropout) for _ in range(3))
        self.attention = nn.ModuleDict({kind: nn.Linear(hidden, 1) for kind in NODE_TYPES if kind != "binary"})
        self.pool_projection = nn.ModuleDict({kind: nn.Linear(hidden, 128) for kind in NODE_TYPES if kind != "binary"})
        self.graph_projection = nn.Sequential(nn.Linear(hidden + 5 * 128, 512), nn.GELU(), nn.Dropout(dropout))

    def encode_nodes(self, graph, static_values, static_available, enc):
        """Create differentiable features from raw metadata, without detaching."""
        x = {"binary": self.static(static_values[None, :], static_available[None]) +
                       self.binary_telemetry(graph["binary"].telemetry.to(enc.device))}
        process_features = enc.process_features(graph["process"].raw)
        x["process"] = torch.cat((process_features[:, :64], self.command_projection(process_features[:, 64:]),
                                  graph["process"].statistics.to(enc.device)), 1)
        apis = graph["api"].raw
        ids = torch.tensor([stable_bucket(api["name"], 8192)[0] for api in apis], device=enc.device, dtype=torch.long)
        categories = []
        for api in apis:
            category_ids = torch.tensor([stable_bucket(category, 128)[0] for category in api["categories"]] or
                                        [stable_bucket("", 128)[0]], device=enc.device)
            categories.append(enc.sequences.category_embedding(category_ids).mean(0))
        category_features = torch.stack(categories) if categories else static_values.new_empty((0, 8))
        flags = static_values.new_tensor([[float(api["imported"]), float(api["resolved"])] for api in apis]).reshape(-1, 2)
        x["api"] = self.api_projection(torch.cat((enc.sequences.api_embedding(ids), category_features, flags), 1))
        x["window"] = self.window_projection(enc.sequences([window["events"] for window in graph["window"].raw], enc.arguments))

        resources = graph["resource"].raw
        groups = {kind: [] for kind in ("file", "registry", "network", "other")}
        for index, resource in enumerate(resources):
            groups[resource["kind"] if resource["kind"] in groups else "other"].append(index)
        resource_features = static_values.new_zeros((len(resources), 128))
        for kind, indices in groups.items():
            if not indices:
                continue
            selected = [resources[index] for index in indices]
            strings = [resource.get("value", "") for resource in selected]
            if kind == "file":
                features = self.file_projection(enc.files(strings))
            else:
                features = enc.other_features(selected) if kind == "other" else getattr(enc, kind)(strings)
                features = self.small_resource_projection[kind](features)
            resource_features = resource_features.index_copy(0, torch.tensor(indices, device=enc.device), features)
        kinds = {"file": 0, "registry": 1, "network": 2, "mutex": 3, "service": 4, "module": 5}
        type_ids = torch.tensor([kinds.get(resource["kind"], 6) for resource in resources], device=enc.device, dtype=torch.long)
        x["resource"] = (resource_features + self.resource_types(type_ids) +
                         self.resource_statistics(graph["resource"].statistics.to(enc.device)))
        x["payload"] = self.payload_projection(enc.payloads(graph["payload"].raw))
        return {kind: self.initial_projection[kind](features) for kind, features in x.items()}

    def forward(self, graphs, static_values, static_available, enc):
        tensor_graphs = []
        for index, graph in enumerate(graphs):
            features = self.encode_nodes(graph, static_values[index], static_available[index], enc)
            tensors = HeteroData()
            for kind in NODE_TYPES:
                tensors[kind].x = features[kind]
            for relation in RELATIONS:
                tensors[relation].edge_index = graph[relation].edge_index.to(enc.device)
            tensor_graphs.append(tensors)
        batch = Batch.from_data_list(tensor_graphs)
        x = batch.x_dict
        for layer in self.layers:
            x = layer(x, batch.edge_index_dict)
        pooled = [x["binary"]]  # Exactly one per graph, in sample order.
        for kind in NODE_TYPES[1:]:
            values = x[kind]
            assignments = batch[kind].batch
            weights = softmax(self.attention[kind](values).squeeze(1), assignments, num_nodes=len(graphs))
            summary = scatter(values * weights[:, None], assignments, dim=0, dim_size=len(graphs), reduce="sum")
            present = torch.bincount(assignments, minlength=len(graphs)) > 0
            summary = self.pool_projection[kind](summary) * present[:, None]
            if kind == "payload":
                coverage = torch.tensor([graph["binary"].raw[0]["payloads_available"] for graph in graphs], device=enc.device)
                summary = summary + enc.payload_coverage(coverage.long())
            pooled.append(summary)
        return self.graph_projection(torch.cat(pooled, 1))


class EvoGraphClassifier(nn.Module):
    """The public list-of-sample classifier; see the module contract."""

    MODES = ("hybrid", "graph", "baseline", "static", "dynamic")

    def __init__(self, mode="hybrid", hidden=256, dropout=.2, window_size=32, max_windows=256):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        # Validate limits even for an ablation that does not construct graphs.
        for name, value, minimum in (("window_size", window_size, 1), ("max_windows", max_windows, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.mode, self.window_size, self.max_windows = mode, window_size, max_windows
        if mode in ("graph", "hybrid"):
            self.encoders = BehavioralEncoders()
            self.graph = GraphEncoder(hidden, dropout)
            if mode == "hybrid":
                self.statistics = nn.Sequential(nn.Linear(STATS_DIM, 64), nn.GELU())
            input_dim = 576 if mode == "hybrid" else 512
        else:
            if mode in ("baseline", "dynamic"):
                self.dynamic = DynamicEncoder()
            if mode in ("baseline", "static"):
                self.static_availability = nn.Embedding(2, STATIC_DIM)
            input_dim = {"baseline": STATIC_DIM + DYNAMIC_DIM, "static": STATIC_DIM, "dynamic": DYNAMIC_DIM}[mode]
        self.head = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, samples):
        device = self.head[0].weight.device
        if not samples:
            return self.head[0].weight.new_empty((0,))
        # Static-only and dynamic-only ablations do not inspect the other input.
        if self.mode != "dynamic":
            static_values = torch.stack([feature_tensor(sample["static_features"], STATIC_DIM, device)
                                         if sample.get("static_features") is not None else
                                         self.head[0].weight.new_zeros(STATIC_DIM) for sample in samples])
            available = torch.tensor([sample.get("static_features") is not None for sample in samples], device=device)
        if self.mode in ("hybrid", "baseline", "dynamic"):
            stats = torch.stack([feature_tensor(sample["statistics"], STATS_DIM, device) for sample in samples])
        if self.mode in ("graph", "hybrid"):
            graphs = [build_graph(sample["report"], self.window_size, self.max_windows) for sample in samples]
            features = self.graph(graphs, static_values, available, self.encoders)
            if self.mode == "hybrid":
                features = torch.cat((features, self.statistics(stats)), 1)
        elif self.mode == "static":
            features = static_values + self.static_availability(available.long())
        else:
            features = torch.stack([self.dynamic(sample["report"], stats[index]) for index, sample in enumerate(samples)])
            if self.mode == "baseline":
                features = torch.cat((static_values + self.static_availability(available.long()), features), 1)
        return self.head(features).squeeze(1)
