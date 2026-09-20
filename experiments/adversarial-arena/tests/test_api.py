import pytest
from fastapi import HTTPException
from app.backend.main import health, runs, start_run, run_path, RunRequest

def test_health_and_real_run_listing():
    assert health()["mode"]=="local_offline"
    listing=runs()
    assert any(x["run_id"]=="smoke_seed20260920_a" for x in listing)

def test_expensive_run_requires_confirmation():
    with pytest.raises(HTTPException) as error:
        start_run(RunRequest(run_id="not_started",confirmed=False))
    assert error.value.status_code==409

def test_path_traversal_rejected():
    with pytest.raises(HTTPException) as error:
        run_path("../secret")
    assert error.value.status_code==400
