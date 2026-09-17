# Extended Attack Families

This framework introduces new advanced attack families and search methods:

- **Motif Replacement Attack** (IMPLEMENTED_NOT_EVALUATED): Replaces atoms with chemically consistent functional groups from a curated library.
- **Scaffold-Preserving Attack** (IMPLEMENTED_NOT_EVALUATED): Constrains edits to peripheral groups, preserving the defined backbone.
- **Evolutionary Search** (IMPLEMENTED_NOT_EVALUATED): Black-box genetic algorithm targeting the property drift.
- **Targeted Objectives** (IMPLEMENTED_NOT_EVALUATED): Allows specifying TARGET_INCREASE, TARGET_DECREASE, or TARGET_VALUE.
- **LLM-Guided Proposals** (PROTOCOL_ONLY): Uses an LLM to propose edits under strict constraint validation. The LLM does NOT judge validity or serve as an oracle.
