import json
from pathlib import Path
from src.materials_adv.robustness.repeat_shift import tokenizer_roundtrip

def test_tokenizer():
    ROOT = Path(".")
    vocabulary = json.loads((ROOT / "data" / "processed" / "vocab.json").read_text())
    
    # Historical behavior
    historical_seq = "C(*)=C"
    hist_check = tokenizer_roundtrip(historical_seq, vocabulary)
    print(f"Historical '{historical_seq}': unknown_count = {hist_check['unknown_count']}")
    
    # Corrected behavior
    corrected_seq = historical_seq.replace("[*]", "__STAR__").replace("*", "[*]").replace("__STAR__", "[*]")
    corr_check = tokenizer_roundtrip(corrected_seq, vocabulary)
    print(f"Corrected '{corrected_seq}': unknown_count = {corr_check['unknown_count']}")
    
    assert corr_check['unknown_count'] == 0, "Tokenizer correction failed"
    print("Tokenizer check passed!")

if __name__ == "__main__":
    test_tokenizer()
