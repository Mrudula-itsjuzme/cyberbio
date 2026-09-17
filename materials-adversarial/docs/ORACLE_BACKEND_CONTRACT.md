# Oracle Backend Contract

This defines the interface constraints all HPC quantum chemistry backends must implement.

## Contract

```python
class OracleBackend(ABC):
    @abstractmethod
    def validate_environment(self) -> EnvironmentStatus:
        pass
        
    @abstractmethod
    def prepare_structure(self, external_structure_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def write_input(self, structure_data: Dict[str, Any], config: Dict[str, Any], output_dir: str) -> None:
        pass
        
    @abstractmethod
    def parse_output(self, output_dir: str) -> ParsedOutput:
        pass

    @abstractmethod
    def validate_result(self, parsed: ParsedOutput) -> bool:
        pass
```

## Immutable Principles

1. **Explicit Uncertainty**: Backends must not invent parameters (cutoffs, functionals). If a parameter is missing, `validate_environment` must return `False` and populate `missing_requirements`.
2. **Explicit Convergence**: Backends must parse their own outputs to classify mathematical convergence distinctly from physical invalidity.
3. **No Silently Discarded Failures**: Backends must retain failed jobs. The ingestion pipeline tracks failure reasons in `ParsedOutput.failure_reason`.
4. **Decoupled Structure Loading**: Backends rely on `structure_data` constructed externally via the structure preparation package. Do not assume wildcards expand deterministically inside the backend integration.
