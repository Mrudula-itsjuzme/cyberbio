# Third-party assets and dataset provenance inventory

This inventory documents known sources and outstanding rights checks. The repository's MIT license applies **only** to original material the contributors have rights to license; it does not relicense external datasets, weights, publications or assets.

| Asset or dataset | Repository path | Known provenance | Rights / redistribution status | Action |
|---|---|---|---|---|
| polyVERSE polymer bandgap | `materials-adversarial/data/raw/bandgap_chain.csv` | Ramprasad-Group/polyVERSE, `Other/bandgap_chain.csv`; Zenodo DOI `10.5281/zenodo.13352644`; see `materials-adversarial/docs/PHASE13A_BANDGAP_PROVENANCE_LOCK.md` | **Not verified by this audit** | Check upstream exact-file license, citation requirements and data redistribution terms |
| Historical literature-sourced polymer properties | `materials-adversarial/data/raw/final_polymer_properties_fromliterature.csv` | Filename indicates literature origin; exact source not verified | **Unverified** | Record DOI(s), extraction method and reuse permissions |
| PI1M reference dataset | `materials-adversarial/data/reference/PI1M.csv` | Reference name PI1M; exact edition and license not verified | **Unverified** | Identify authoritative release, version, attribution and redistribution rights |
| Synthetic bio-cyber datasets | `bio-cyber-adversarial/data/` | Locally generated synthetic sequence benchmarks per subproject README | **Likely project-generated; verify** | Preserve generator, seed and manifest; confirm no external inputs |
| Derived attack banks, processed datasets and results | `materials-adversarial/data/attacks/`, `materials-adversarial/data/processed/`, `results/` | Project-generated transformations of source data | **Dependent on upstream terms** | Identify source lineage for each public export; avoid claiming MIT applies to underlying data |
| Models, checkpoints and other research artifacts | Various experiment and result directories | Produced or bundled for experiments; not individually verified | **Unverified per artifact** | Record training sources, model origin and relevant third-party terms |

## Reuse boundary
All entries marked unverified are **not cleared for redistribution** by this inventory. The repository-level MIT license does not supersede dataset or checkpoint rights. Cite upstream sources and check terms before using or redistributing these assets.

## Provenance quality
This file records a targeted repository inspection, not a legal clearance or a complete per-file trace. Revisit after validating upstream project license files and exact dataset sources.

## Verified upstream license (2026-10-31)
The polyVERSE upstream `LICENSE` in `Ramprasad-Group/polyVERSE`, corroborated by Zenodo release DOI `10.5281/zenodo.13352644`, is the **Georgia Tech Research Corporation General Public Use License Agreement**, **not MIT or a standard open-source license**. Its terms allow copies with copyright, license and warranty notices kept intact and a license copy accompanying distributions, while restricting sale for commercial gain without a separate GTRC agreement. These are upstream terms, not permission to apply CyberBio's MIT license to its data.

Sources:
- https://github.com/Ramprasad-Group/polyVERSE/blob/main/LICENSE
- https://zenodo.org/records/13352644

**Action required:** Verify the exact source and version of each bundled polyVERSE-derived file. Where upstream license applies, include the required original license and notices alongside the redistributed material, or stop redistributing pending verification. **Not yet cleared** for full compliance. The PI1M and literature-derived datasets remain unresolved.
