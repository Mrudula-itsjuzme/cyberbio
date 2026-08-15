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


def test_analysis_layers_do_not_pull_in_torch() -> None:
    """The tokenizer, attack-mechanics, validation and evaluation layers must stay
    importable without torch.

    Originally this asserted the whole package was torch-free, which was correct
    when no model existed. models/transformer.py now legitimately imports torch,
    so the check is scoped to the layers where the property still holds and still
    matters: they are pure string/array logic and should remain analysable
    without a deep-learning stack.

    Run in a subprocess -- in-process checks are polluted by other tests.
    """
    code = textwrap.dedent(
        """
        import sys
        from materials_adv.data import tokenizer            # noqa: F401
        from materials_adv.attacks import token_space, base, deletion  # noqa: F401
        from materials_adv.validation import pipeline, representation, plausibility  # noqa: F401
        from materials_adv.evaluation import records, metrics, attack_metrics  # noqa: F401
        assert "torch" not in sys.modules, "an analysis module imported torch at top level"
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


def test_preprocessing_is_implemented() -> None:
    """Was a stub-raises test. Preprocessing was implemented once the data arrived.

    The stub contract no longer applies; what matters now is that the real
    entry point exists and is callable.
    """
    from materials_adv.data import preprocessing

    assert callable(preprocessing.build_processed_dataset)
    assert callable(preprocessing.canonicalize)


def test_training_and_model_entry_points_are_implemented() -> None:
    """Was a stub-raises test.

    transformer.build_transformer_regressor no longer exists -- the Phase 1
    implementation exposes the TransformerRegressorModel class directly.
    """
    from materials_adv.models import regression, transformer
    from materials_adv.training import train

    assert hasattr(transformer, "TransformerRegressorModel")
    assert callable(train.train)
    assert hasattr(regression, "TransformerRegressor")
    assert not hasattr(transformer, "build_transformer_regressor")


def test_train_accepts_the_phase2b_scaler_control() -> None:
    """Regression: the fixed-scaler control must not be silently dropped."""
    import inspect

    from materials_adv.training.train import train

    params = inspect.signature(train).parameters
    assert "scaler_path" in params
    assert "write_back_config" in params
    assert "seed" in params
    # Defaults must not silently re-enable the Phase 2A confounder or the
    # config-overwrite bug for callers that pass neither.
    assert params["scaler_path"].default is None
    assert params["seed"].default is None


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


def test_shipped_configs_are_resolved_after_the_audit() -> None:
    """Was 'PENDING values are null'. The audit resolved them, so the contract
    inverted: these keys must now be POPULATED, and a regression to null would
    mean a config was clobbered.
    """
    from materials_adv.utils.config import get, load_config

    dataset = load_config("dataset")
    assert get(dataset, "representation_column") == "PSMILES"
    assert get(dataset, "target_column") is not None
    assert get(dataset, "target_units") is not None
    assert get(dataset, "split.test_sealed") is True

    model = load_config("model")
    assert get(model, "architecture.d_model") is not None
    assert get(model, "architecture.n_layers") is not None
    assert get(model, "device") == "cpu"

    attack = load_config("attack")
    assert get(attack, "protection.protect_attachments") is True
    assert get(attack, "protection.protect_ring_closures") is True


def test_split_fractions_still_sum_to_one() -> None:
    """The split artifact is frozen; its config must not silently drift from it."""
    from materials_adv.utils.config import get, load_config

    cfg = load_config("dataset")
    total = (
        get(cfg, "split.train_frac")
        + get(cfg, "split.val_frac")
        + get(cfg, "split.test_frac")
    )
    assert abs(total - 1.0) < 1e-9


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
