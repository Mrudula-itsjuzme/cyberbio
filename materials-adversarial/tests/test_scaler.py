import pytest
import numpy as np
from pathlib import Path
from materials_adv.data.scaler import TargetScaler

def test_scaler_fit_transform():
    y_train = np.array([100.0, 200.0, 300.0])
    scaler = TargetScaler()
    scaler.fit(y_train)
    
    assert scaler.mean == 200.0
    assert scaler.std == pytest.approx(81.64965809277261)
    
    y_norm = scaler.transform(y_train)
    np.testing.assert_allclose(y_norm, np.array([-1.22474487, 0.0, 1.22474487]))
    
    y_inv = scaler.inverse_transform(y_norm)
    np.testing.assert_allclose(y_train, y_inv)

def test_scaler_test_set_unseen():
    y_train = np.array([100.0, 200.0, 300.0])
    y_test = np.array([400.0])
    
    scaler = TargetScaler()
    scaler.fit(y_train)
    
    y_test_norm = scaler.transform(y_test)
    y_test_inv = scaler.inverse_transform(y_test_norm)
    
    # Assert fitting didn't use test
    assert scaler.mean == 200.0
    assert scaler.inverse_transform(scaler.transform(400.0)) == pytest.approx(400.0)

def test_scaler_save_load(tmp_path):
    y_train = np.array([150.0, 350.0])
    scaler = TargetScaler()
    scaler.fit(y_train)
    
    save_path = tmp_path / "scaler.json"
    scaler.save(save_path)
    
    scaler2 = TargetScaler.load(save_path)
    assert scaler2.mean == scaler.mean
    assert scaler2.std == scaler.std

def test_scaler_constant():
    scaler = TargetScaler()
    scaler.fit(np.array([100.0, 100.0]))
    assert scaler.mean == 100.0
    assert scaler.std == 1.0 # Should not be 0
    assert scaler.transform(100.0) == 0.0
    assert scaler.inverse_transform(0.0) == 100.0
