# EvoGraph Methodology Note

`evograph_methodology.tex` is a short, standalone IEEE Transactions-style
methodology note using `\documentclass[journal]{IEEEtran}` and the neutral author
`EvoGraph Project`. It is not an IEEE submission or an accepted publication.
The target is at most three two-column pages, approximately 1,000-1,400 words
including references, with one compact representation table and inline equations.

## Build

Run from this directory with an existing LaTeX installation containing `IEEEtran`,
`amsmath`, `amssymb`, `booktabs`, and `url`:

```sh
latexmk -pdf evograph_methodology.tex
```

Alternatively, run twice:

```sh
pdflatex evograph_methodology.tex
pdflatex evograph_methodology.tex
```

No external figures, bibliography files, or BibTeX step are required. The included
`evograph_methodology.pdf` was compiled and checked as **two pages** using Tectonic
0.17.0. An alternative build is `tectonic evograph_methodology.tex`; its first
run may download the required TeX packages. Pagination can vary between engines.

## Scope

The authoritative sources are the current [`multimodal/`](../multimodal/)
implementation and [`MULTIMODAL.md`](../MULTIMODAL.md). The note focuses on
evidence-aware integration, six node types, source-order windows, typed resource
interactions, missingness, exact relation-mean/residual updates, type attention,
and shared-split flat/graph comparisons. It distinguishes the 4,429-dimensional
flat input from the hybrid's graph512 plus statistics64 input.

There is no trained research model or full multimodal benchmark. Tests and smoke
checks establish plumbing only; the note contains no empirical scores, empty
results table, superiority claim, or claim of a new GNN operator. CAPE reports
are user-supplied; sandbox execution and API transport are outside scope. HGT and
incremental learning are deferred.

## Extraction Provenance

The recorded locked, isolated extraction profile is official Elastic EMBER
revision `d97a0b523de02f3fe5ea6089d080abacab6ee931`, Python `3.11.14`, NumPy
`1.23.5`, scikit-learn `1.1.3`, and LIEF `0.12.3`. It is not an exact replication
of canonical EMBER2018 extraction with LIEF `0.9.0`. Record this provenance with
extracted features and use a consistent profile for training and inference.
The original Anderson/Roth paper describes feature version 1; the official
repository documents feature version 2 used here.
