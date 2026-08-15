"""Guard tests: the scaffold must stay importable, and stubs must stay loud.

These catch two regressions that would otherwise pass silently:
  1. someone adds a top-level `import torch`, breaking the whole package here
  2. a PENDING stub quietly starts returning something instead of raising
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap

import pytest

from materials_adv.utils.pending import PendingImplementation

PACKAGE_MODULES = [
    "materials_adv",
    "materials_adv.attacks.base",
    "materials_adv.attacks.deletion",
    "materials_adv.attacks.generator",
    "materials_adv.attacks.insertion",
    "materials_adv.attacks.registry",
    "materials_adv.attacks.reordering",
    "materials_adv.attacks.substitution",
    "materials_adv.attacks.token_space",
    "materials_adv.data.loader",
    "materials_adv.data.preprocessing",
    "materials_adv.data.splits",
    "materials_adv.data.tokenizer",
    "materials_adv.evaluation.attack_metrics",
    "materials_adv.evaluation.metrics",
    "materials_adv.evaluation.records",
    "materials_adv.models.regression",
    "materials_adv.models.registry",
    "materials_adv.models.transformer",
    "materials_adv.training.evaluate",
    "materials_adv.training.train",
    "materials_adv.utils.config",
    "materials_adv.utils.io",
    "materials_adv.utils.logging",
    "materials_adv.utils.optional",
    "materials_adv.utils.pending",
    "materials_adv.validation.pipeline",
    "materials_adv.validation.plausibility",
    "materials_adv.validation.representation",
]


@pytest.mark.parametrize("module", PACKAGE_MODULES)
def test_every_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_importing_package_does_not_pull_in_torch() -> None:
    """Run in a subprocess: in-process checks are polluted by other tests."""
    code = textwrap.dedent(
        """
        import sys
        import materials_adv
        from materials_adv.attacks import generator, substitution  # noqa: F401
        from materials_adv.validation import pipeline  # noqa: F401
        from materials_adv.models import transformer, registry  # noqa: F401
        assert "torch" not in sys.modules, "a module imported torch at top level"
        print("clean")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# --- Stubs must raise, never return ------------------------------------------


def test_loader_stubs_raise() -> None:
    from materials_adv.data import loader

    with pytest.raises(PendingImplementation):
        loader.load_raw("anything.csv")
    with pytest.raises(PendingImplementation):
        loader.load_property_subset("anything.csv", "Tg")


def test_preprocessing_stub_raises() -> None:
    from materials_adv.data import preprocessing

    with pytest.raises(PendingImplementation):
        preprocessing.build_processed_dataset()


def test_model_and_training_stubs_raise() -> None:
    from materials_adv.models import regression, transformer
    from materials_adv.training import evaluate, train

    for call in (
        transformer.build_transformer_regressor,
        train.train,
        evaluate.evaluate,
        regression.TransformerRegressor,
    ):
        with pytest.raises(PendingImplementation):
            call()


def test_pending_error_names_its_blocker_and_remedy() -> None:
    from materials_adv.data import loader

    with pytest.raises(PendingImplementation) as exc:
        loader.load_raw("x.csv")
    assert exc.value.blocked_on == "dataset"
    assert "audit_dataset" in exc.value.unblocks_when


# --- Config null-safety ------------------------------------------------------


def test_require_resolved_raises_on_pending_null() -> None:
    from materials_adv.utils.config import require_resolved

    with pytest.raises(PendingImplementation):
        require_resolved({"a": {"b": None}}, "a.b", unblocks_when="the audit runs")


def test_get_returns_nested_value() -> None:
    from materials_adv.utils.config import get

    assert get({"a": {"b": 3}}, "a.b") == 3
    assert get({"a": {}}, "a.missing", default="fallback") == "fallback"


def test_shipped_configs_load_and_mark_pending_as_null() -> None:
    from materials_adv.utils.config import get, load_config

    dataset = load_config("dataset")
    assert get(dataset, "representation_column") is None
    assert get(dataset, "target_units") is None
    assert get(dataset, "split.test_sealed") is True

    model = load_config("model")
    assert get(model, "architecture.d_model") is None
    assert get(model, "device") == "cpu"

    attack = load_config("attack")
    assert get(attack, "attacks.substitution.allowed_tokens") is None
    assert get(attack, "protection.protect_attachments") is True


# --- Seeding -----------------------------------------------------------------


def test_seeding_is_reproducible_across_processes() -> None:
    """Subprocess, because in-process global RNG state hides bugs."""
    code = textwrap.dedent(
        """
        import numpy as np
        from materials_adv.utils.seeding import seed_everything, make_rng
        seed_everything(123)
        print(np.random.rand(), make_rng(123).random())
        """
    )
    runs = [
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        for _ in range(2)
    ]
    assert runs[0].stdout == runs[1].stdout


def test_seed_record_reports_torch_honestly() -> None:
    from materials_adv.utils.optional import has_torch
    from materials_adv.utils.seeding import seed_everything

    assert seed_everything(1)["torch"] == has_torch()


def test_make_rng_is_independent_of_global_state() -> None:
    import numpy as np

    from materials_adv.utils.seeding import make_rng

    np.random.seed(999)
    a = make_rng(7).random()
    np.random.seed(111)
    assert make_rng(7).random() == a
