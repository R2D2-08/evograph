"""Trainable EMBFV encoders. Inputs are parse_report IR and scaled statistics.

String/token bucketing uses fixed BLAKE2b, never Python's salted hash. Only D2
is fixed feature hashing; embeddings, convolutions, recurrent networks and
pooling are ordinary trainable modules. No report-dependent tensors are cached.
"""

import hashlib
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from .cape import canonical_api
from .graph import event_streams
from .schema import DYNAMIC_DIM, STATIC_DIM, STATS_DIM


def stable_bucket(value, buckets=512):
    """Return a stable (bucket, sign), with independent digest bytes for sign."""
    if buckets < 1:
        raise ValueError("buckets must be positive")
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest[:8], "little") % buckets, 1 if digest[8] & 1 else -1


def api_frequencies(report, device=None):
    """D2: signed counts divided by total executed occurrences, shape [512]."""
    counts = Counter()
    for process in report.get("processes", []):
        for name, count in process.get("api_counts", {}).items():
            api = canonical_api(name)
            if api and count > 0:
                counts[api] += count
    result = torch.zeros(512, device=device)
    total = sum(counts.values())
    if total:
        for api, count in sorted(counts.items()):
            bucket, sign = stable_bucket(api)
            result[bucket] += sign * (count / total)
    return result


def meanmax(values):
    """Mean/max over rows, with a zero vector for an empty group."""
    if not values.size(0):
        return values.new_zeros(values.size(1) * 2)
    return torch.cat((values.mean(0), values.max(0).values))


def feature_tensor(values, dimension, device):
    """Validate supplied numerical vectors without silently imputing bad data."""
    result = torch.as_tensor(values, dtype=torch.float32, device=device)
    if result.shape != (dimension,) or not torch.isfinite(result).all():
        raise ValueError(f"expected a finite vector of shape ({dimension},)")
    return result


def normalize_user_path(path):
    """Mask only a local Windows profile name, preserving the rest of a path.

    Shared/default profiles retain their semantic names. Only paths descending
    into a profile are normalized, so files directly in Users stay filenames.
    UNC shares and relative paths are not guessed to be user profiles.
    Normalization is encoder-only; report strings and graph identity stay raw.
    """
    match = re.match(r"(?i)^([a-z]:[\\/](?:users|documents and settings)[\\/])([^\\/]+)(?=[\\/])", str(path))
    if not match or match[2].lower() in {"public", "default", "default user", "all users", ".", ".."}:
        return str(path)
    return match[1] + "<USER>" + str(path)[match.end():]


class CharCNN(nn.Module):
    """UTF-8 byte CNN with masked pooling, independent of batch padding."""

    def __init__(self, output_dim=64, max_length=256, normalize_paths=False):
        super().__init__()
        self.output_dim = output_dim
        self.max_length = max_length
        self.normalize_paths = normalize_paths
        self.embedding = nn.Embedding(257, 16, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(16, 32, kernel, padding=kernel // 2) for kernel in (3, 5, 7))
        self.projection = nn.Linear(96, output_dim)

    def forward(self, strings):
        device = self.embedding.weight.device
        if not strings:
            return self.embedding.weight.new_empty((0, self.output_dim))
        if self.normalize_paths:
            strings = [normalize_user_path(text) for text in strings]
        rows = [[byte + 1 for byte in str(text).encode("utf-8")[:self.max_length]] or [0] for text in strings]
        lengths = torch.tensor([len(row) for row in rows], device=device)
        tokens = pad_sequence([torch.tensor(row, device=device) for row in rows], batch_first=True)
        mask = torch.arange(tokens.size(1), device=device)[None, :] < lengths[:, None]
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [torch.relu(conv(embedded)).masked_fill(~mask[:, None, :], -torch.inf).amax(2)
                  for conv in self.convs]
        return self.projection(torch.cat(pooled, 1))


def normalize_network_identifier(value):
    """Mask literal IP hosts, not domains or URL paths, before string encoding.

    IPv4/IPv6 identities become distinct kind markers. IP endpoint ports retain
    only their presence; schemes, userinfo, paths, queries and fragments stay
    unchanged. Bare IPv6 is an address, not an invented address/port split.
    Invalid/ambiguous identifiers are left alone, without DNS resolution.
    """
    text = str(value)
    try:
        return f"<IPV{ipaddress.ip_address(text).version}>"
    except ValueError:
        pass
    source = text if "://" in text or text.startswith("//") else "//" + text
    try:
        parsed = urlsplit(source)
        address = ipaddress.ip_address(parsed.hostname or "")
    except ValueError:
        return text
    start = source.index("//") + 2
    # urlsplit can discard control characters; never splice a changed authority.
    if source[start:start + len(parsed.netloc)] != parsed.netloc:
        return text
    userinfo, at, endpoint = parsed.netloc.rpartition("@")
    marker = f"<IPV{address.version}>"
    if endpoint.startswith("["):
        suffix = endpoint[endpoint.index("]") + 1:]
        if suffix and not suffix.startswith(":"):
            return text
        masked = f"[{marker}]" + (":<PORT>" if suffix else "")
    else:
        masked = marker + (":<PORT>" if ":" in endpoint else "")
    result = source[:start] + userinfo + at + masked + source[start + len(parsed.netloc):]
    return result if source == text else result[2:]


class NetworkEncoder(CharCNN):
    """Shared D7/graph character encoder with literal destination IP masking."""

    def forward(self, strings):
        return super().forward([normalize_network_identifier(text) for text in strings])


class CommandEncoder(nn.Module):
    """Order-sensitive command token encoder, with a bounded stable vocabulary."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4097, 32, padding_idx=0)
        self.gru = nn.GRU(32, 32, batch_first=True, bidirectional=True)

    def forward(self, commands):
        if not commands:
            return self.embedding.weight.new_empty((0, 64))
        rows = [[1 + stable_bucket(token.lower(), 4096)[0]
                 for token in re.findall(r'"[^"\n]*"|\S+', str(command))[:64]] or [0]
                for command in commands]
        device = self.embedding.weight.device
        tokens = pad_sequence([torch.tensor(row, device=device) for row in rows], batch_first=True)
        packed = pack_padded_sequence(self.embedding(tokens), [len(row) for row in rows], batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return torch.cat((hidden[-2], hidden[-1]), 1)


class RegistryEncoder(nn.Module):
    """Character-encoded path components followed by a hierarchy BiLSTM."""

    def __init__(self):
        super().__init__()
        self.components = CharCNN(32, max_length=64)
        self.lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)

    def forward(self, paths):
        if not paths:
            return self.components.embedding.weight.new_empty((0, 64))
        parts = [[part for part in re.split(r"[\\/]", str(path)) if part][:32] or [""] for path in paths]
        embeddings = self.components([part for row in parts for part in row])
        sequences = embeddings.split([len(row) for row in parts])
        packed = pack_padded_sequence(pad_sequence(sequences, batch_first=True), [len(row) for row in parts],
                                      batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return torch.cat((hidden[-2], hidden[-1]), 1)


class TypedArgumentEncoder(nn.Module):
    """Encode types, numerical sizes/flags and known semantics, not raw handles.

    Layout: 8 type counts, 8 argument-role counts, 32 numerical flag bits,
    8 symbolic flag families, 6 resource-kind counts, 3 status flags, and
    argument count / signed-log size mean / maximum (68 inputs). Unknown text,
    pointer values, PIDs and handle magnitudes are deliberately discarded.
    """

    RESOURCE_KINDS = ("file", "registry", "network", "mutex", "service", "module")
    FLAG_FAMILIES = ("read", "write", "execute", "all_access", "commit", "reserve", "create", "open")

    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(68, 96), nn.GELU(), nn.Linear(96, 64))

    @staticmethod
    def semantics(event):
        features = [0.0] * 68
        sizes = []
        arguments = event.get("arguments", [])
        for argument in arguments:
            name = re.sub(r"[^a-z0-9]", "", argument.get("name", "").lower())
            value = argument.get("value")
            type_index = (0 if value is None else 1 if isinstance(value, bool) else
                          2 if isinstance(value, int) else 3 if isinstance(value, float) else
                          4 if isinstance(value, str) else 5 if isinstance(value, list) else
                          6 if isinstance(value, dict) else 7)
            features[type_index] += 1
            role = (0 if "handle" in name or name.startswith("hprocess") else
                    1 if any(part in name for part in ("pointer", "address", "buffer", "baseaddr")) else
                    2 if any(part in name for part in ("size", "length", "count", "bytes")) else
                    3 if any(part in name for part in ("protect", "access", "flag", "mode", "type", "disposition", "option")) else
                    4 if any(part in name for part in ("path", "file", "key", "name", "host", "url")) else
                    5 if "port" in name else 6 if "id" in name else 7)
            features[8 + role] += 1
            number = None
            if role in (2, 3, 5):
                try:
                    number = int(value, 0) if isinstance(value, str) else float(value)
                    if not math.isfinite(number):
                        number = None
                except (TypeError, ValueError, OverflowError):
                    number = None
            if role == 3:
                if number is not None:
                    bits = int(number) & 0xFFFFFFFF
                    for bit in range(32):
                        features[16 + bit] += (bits >> bit) & 1
                symbolic = str(argument.get("pretty_value") or value or "").lower()
                for index, family in enumerate(TypedArgumentEncoder.FLAG_FAMILIES):
                    features[48 + index] += family in symbolic
            if role in (2, 5) and number is not None:
                sizes.append(math.copysign(math.log1p(abs(number)), number))
        for resource in event.get("resources", []):
            if resource.get("kind") in TypedArgumentEncoder.RESOURCE_KINDS:
                features[56 + TypedArgumentEncoder.RESOURCE_KINDS.index(resource["kind"])] += 1
        status = event.get("status")
        features[62 + (0 if status is True else 1 if status is False else 2)] = 1
        features[65:] = [math.log1p(len(arguments)), sum(sizes) / len(sizes) if sizes else 0,
                         max(sizes, default=0)]
        # Bound argument/resource count features without touching flag semantics.
        for index in list(range(16)) + list(range(56, 62)):
            features[index] = math.log1p(features[index])
        return features

    def forward(self, events):
        device = self.projection[0].weight.device
        values = torch.tensor([self.semantics(event) for event in events], device=device, dtype=torch.float32).reshape(-1, 68)
        return self.projection(values)


class APISequenceEncoder(nn.Module):
    """Per-stream BiGRU + event attention, with stream attention for D3.

    Time inputs are signed log gaps/offsets within the same stream and clock
    domain, plus availability bits. Absolute host/guest timestamps and raw
    thread identifiers never enter embeddings. Backward gaps stay negative.
    """

    def __init__(self):
        super().__init__()
        self.api_embedding = nn.Embedding(8192, 64)
        self.category_embedding = nn.Embedding(128, 8)
        self.gru = nn.GRU(144, 128, batch_first=True, bidirectional=True)
        self.event_attention = nn.Linear(256, 1)
        self.stream_attention = nn.Linear(256, 1)

    def forward(self, streams, argument_encoder):
        device = self.api_embedding.weight.device
        if not streams:
            return self.api_embedding.weight.new_empty((0, 256))
        flattened, metadata, lengths = [], [], []
        for events in streams:
            if not events:
                raise ValueError("sequence streams must be nonempty")
            first, previous = {}, {}
            lengths.append(len(events))
            for event in events:
                timestamp, domain = event.get("timestamp"), event.get("timestamp_domain", "numeric")
                valid = isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool) and math.isfinite(timestamp)
                gap_available = valid and domain in previous
                gap = timestamp - previous[domain] if gap_available else 0
                offset = timestamp - first.setdefault(domain, timestamp) if valid else 0
                if valid:
                    previous[domain] = timestamp
                signed_log = lambda value: math.copysign(math.log1p(abs(value)), value) if math.isfinite(value) else 0.0
                status = event.get("status")
                metadata.append([signed_log(gap), signed_log(offset), float(valid), float(gap_available),
                                 float(status is True), float(status is False), float(status is None),
                                 math.log1p(max(0, event.get("repeat", 1)))])
                flattened.append(event)
        api_ids = torch.tensor([stable_bucket(canonical_api(event.get("api")), 8192)[0] for event in flattened], device=device)
        category_ids = torch.tensor([stable_bucket(event.get("category", ""), 128)[0] for event in flattened], device=device)
        inputs = torch.cat((self.api_embedding(api_ids), self.category_embedding(category_ids),
                            argument_encoder(flattened), torch.tensor(metadata, device=device)), 1)
        packed = pack_padded_sequence(pad_sequence(inputs.split(lengths), batch_first=True), lengths,
                                      batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        mask = torch.arange(output.size(1), device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
        weights = self.event_attention(output).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(1)
        return (output * weights[:, :, None]).sum(1)

    def pool_streams(self, encoded):
        if not encoded.size(0):
            return encoded.new_zeros(256)
        return (encoded * self.stream_attention(encoded).softmax(0)).sum(0)


class PayloadEncoder(nn.Module):
    """2381 -> 64, with explicit available/missing learned embeddings."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(STATIC_DIM, 64)
        self.availability = nn.Embedding(2, 64)

    def forward(self, payloads):
        device = self.projection.weight.device
        if not payloads:
            return self.projection.weight.new_empty((0, 64))
        available = torch.tensor([payload.get("static_features") is not None for payload in payloads], device=device)
        values = torch.stack([feature_tensor(payload["static_features"], STATIC_DIM, device)
                              if payload.get("static_features") is not None else self.projection.weight.new_zeros(STATIC_DIM)
                              for payload in payloads])
        return self.projection(values) * available[:, None] + self.availability(available.long())


class BehavioralEncoders(nn.Module):
    """One shared set of trainable string, sequence, argument and payload modules."""

    def __init__(self):
        super().__init__()
        self.files = CharCNN(128, normalize_paths=True)
        self.process_names = CharCNN(64)
        self.commands = CommandEncoder()
        self.registry = RegistryEncoder()
        self.network = NetworkEncoder(64)
        self.other = CharCNN(64)
        self.other_types = nn.Embedding(4, 64)
        self.arguments = TypedArgumentEncoder()
        self.sequences = APISequenceEncoder()
        self.payloads = PayloadEncoder()
        self.payload_coverage = nn.Embedding(2, 128)

    @property
    def device(self):
        return self.files.embedding.weight.device

    def process_features(self, processes):
        return torch.cat((self.process_names([p.get("name", "") for p in processes]),
                          self.commands([p.get("command_line", "") for p in processes])), 1)

    def other_features(self, resources):
        ids = [{"mutex": 0, "service": 1, "module": 2}.get(item.get("kind"), 3) for item in resources]
        return self.other([item.get("value", "") for item in resources]) + self.other_types(torch.tensor(ids, device=self.device, dtype=torch.long))


class DynamicEncoder(nn.Module):
    """``forward(report, statistics) -> [2048]`` in frozen schema block order.

    Statistics must already be standardized by the caller. D3 encodes each
    retained process/thread stream independently, then attention-pools streams;
    it does not fabricate sequences from counts. Graph window caps do not
    truncate this baseline branch (the parser's event-retention limit applies).
    """

    def __init__(self, encoders=None):
        super().__init__()
        self.encoders = encoders if encoders is not None else BehavioralEncoders()

    def forward(self, report, statistics):
        enc = self.encoders
        stats = feature_tensor(statistics, STATS_DIM, enc.device)
        by_kind = {kind: [] for kind in ("file", "registry", "network")}
        other = []
        for resource in report.get("resources", []):
            (by_kind[resource["kind"]] if resource.get("kind") in by_kind else other).append(resource)
        processes = report.get("processes", [])
        process_features = enc.process_features(processes)
        streams = [[event for _, event in events] for _, _, events in event_streams(report)]
        events = [event for process in processes for event in process.get("events", [])]
        payloads = meanmax(enc.payloads(report.get("payloads", [])))
        coverage = int(bool(report.get("coverage", {}).get("payloads")))
        payloads = payloads + enc.payload_coverage.weight[coverage]
        blocks = [stats, api_frequencies(report, enc.device),
                  enc.sequences.pool_streams(enc.sequences(streams, enc.arguments)),
                  meanmax(enc.files([item.get("value", "") for item in by_kind["file"]])),
                  meanmax(process_features[:, :64]), meanmax(process_features[:, 64:]),
                  meanmax(enc.registry([item.get("value", "") for item in by_kind["registry"]])),
                  meanmax(enc.network([item.get("value", "") for item in by_kind["network"]])),
                  meanmax(enc.other_features(other)), meanmax(enc.arguments(events)), payloads]
        result = torch.cat(blocks)
        assert result.shape == (DYNAMIC_DIM,)
        return result
