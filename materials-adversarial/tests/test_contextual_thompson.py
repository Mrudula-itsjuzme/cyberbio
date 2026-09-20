import numpy as np
from materials_adv.attacks.search.context import ContextFeatureExtractor, ContextScaler

def test_context_extractor_deterministic():
    vocab = ["C", "O", "N", "c", "1", "2", "(", ")", "="]
    extractor = ContextFeatureExtractor(vocab)
    
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    f1 = extractor.extract(smiles)
    f2 = extractor.extract(smiles)
    
    np.testing.assert_array_almost_equal(f1, f2)
    assert len(f1) == extractor.dim

def test_context_scaler_no_leakage():
    scaler = ContextScaler()
    train_features = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    scaler.fit(train_features)
    
    # Validation/Test features must not change the scaler state
    mean_before = np.copy(scaler.mean)
    std_before = np.copy(scaler.std)
    
    test_features = np.array([10.0, 20.0])
    scaled = scaler.transform(test_features)
    
    np.testing.assert_array_almost_equal(scaler.mean, mean_before)
    np.testing.assert_array_almost_equal(scaler.std, std_before)
