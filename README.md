# Shock-ALE-SW

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.11-blue.svg)](https://www.python.org)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19570235.svg)](https://doi.org/10.5281/zenodo.19570235) 

## Description

Shock-ALE-SW is an openly available Python software package for two-dimensional shallow-water shock-wave simulations on moving triangular meshes. The package implements a Godunov-type finite-volume method with ALE mesh motion, exact local Riemann-state evaluation, and a rotated Riemann solver. The repository also includes a reproducible notebook for a single planar shock-wave benchmark in a circular domain.

![Single planar shock on a moving triangular mesh](demo.gif)

## Registration

Certificate of state registration of a computer program No. 2026660158 dated 09.04.2026 (application No. 2026619222 dated 02.04.2026).

## Authors and Right Holder

- Authors: Stanislav A. Ladygin, Kirill E. Shilnikov
- Right holder: National Research Nuclear University MEPhI (NRNU MEPhI)

## Funding

This work was supported by the Russian Science Foundation (RSF), project No. 24-71-00113, "Development of numerical methods for problems related to hyperbolic systems of partial differential equations on dynamically adaptive meshes" (July 2024 to June 2026). Principal investigator: Kirill E. Shilnikov. Host organization: National Research Nuclear University MEPhI (NRNU MEPhI), Moscow.

## Installation

Install the core package from the repository root:

```bash
python -m pip install .
```

Install the optional notebook environment:

```bash
python -m pip install ".[notebook]"
```

## Usage

Run the demonstration notebook:

```bash
jupyter lab notebook_single_shock.ipynb
```

The public Python API is intentionally small:

```python
from shock_only_sw import ShockMesh, run_simulation, compute_l1_errors
```

## Repository Structure

```text
shock_only_sw/            Core package
notebook_single_shock.ipynb
pyproject.toml            Packaging metadata
README.md                 Public project description
LICENSE                   Apache License 2.0
NOTICE                    Attribution and funding notice
CITATION.cff              Citation metadata for GitHub and reference managers
CONTRIBUTING.md           Contribution guidelines
```

## Citation

Please use the metadata in `CITATION.cff` when citing this software. GitHub can expose the citation entry directly from that file after publication.

## License

This project is distributed under the Apache License 2.0. See `LICENSE` for the full license text and `NOTICE` for attribution information.
