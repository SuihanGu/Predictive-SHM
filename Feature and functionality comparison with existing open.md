Feature and functionality comparison with existing open-source tools

**Table 1. Compared systems**

| Project | Reference |
| - | - |
| Predictive-SHM | https://github.com/SuihanGu/Predictive-SHM |
| OpenBDLM | https://github.com/CivML-PolyMtl/OpenBDLM |
| pyOMA | https://github.com/simonmarwitz/pyOMA |
| MIDAS-SHM | https://github.com/human-analysis/midas-shm |


Table 2. Feature comparison

| Feature | Predictive-SHM | OpenBDLM | pyOMA | MIDAS-SHM |
| - | - | - | - | - |
| Primary intent | End-to-end field-style monitoring stack | BDLM-centric long-horizon time-series modelling & anomaly detection for SHM | Operational modal analysis  | Damage assessment via mechanics-informed ML  |
| Typical stack | Vue 3 + FastAPI, REST/JSON | MATLAB + toolboxes | Python  | Python |
| Browser-based monitoring UI | Yes  | No  | No | No  |
| REST/HTTP ingest API (reference release) | Yes  | No native HTTP service in core release | No | No |
| Built-in alignment for async sensors  | Yes  | Partial  | N/A | N/A |
| Unified logical schema  | Yes (model\_config.json + ULDM) | No (BDLM state–space formulation, not PS ULDM) | No | No |
| Configuration-driven pluggable forecasters  | Yes  | No  | No | No  |
| Threshold / residual alerting in stack | Yes  | No  | No | No  |
| Dedicated time-series DB required | No  | No | No | No |
| Open license  | MIT | MIT | GPL | GitHub listsOther — confirm with authors |

Table 1 provides a qualitative comparison of representative open-source SHM projects from two perspectives: tool positioning and deployable form. Predictive-SHM (v1.0) targets field‑deployment scenarios by offering an end‑to‑end browser‑based monitoring stack that encompasses data ingestion, visualization, multi‑step prediction, and lightweight alerting. It uses the ULDM to unify data column ordering and roles, and supports pluggable extension of prediction models through a registry‑plus‑adapter mechanism. In contrast, OpenBDLM focuses more on BDLM‑based probabilistic modeling and anomaly detection (primarily through MATLAB workflows), pyOMA concentrates on operational modal analysis, and MIDAS‑SHM centers on mechanics‑constrained damage assessment research workflows. It should be emphasized that the entries in the table (e.g., “lightweight deployment,” “partial support”) represent qualitative judgments: Predictive‑SHM, in its reference release, deliberately avoids mandating dedicated time‑series databases or clusters so as to lower the operational barrier for demonstrations and reproducibility experiments. Other projects, while they may not require server clusters, may still incur different forms of dependency costs related to desktop runtime environments, commercial runtimes (e.g., MATLAB), or computational resources (e.g., GPUs).

