import os
import hashlib
import json
from glob import glob

def get_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def create_hashes():
    bundle_dir = "oracle_surrogate_qc_bundle"
    hashes = {}
    
    for root, dirs, files in os.walk(bundle_dir):
        for file in files:
            if file == "HASHES.json":
                continue
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, bundle_dir)
            hashes[rel_path] = get_hash(filepath)
            
    with open(os.path.join(bundle_dir, "HASHES.json"), "w") as f:
        json.dump(hashes, f, indent=4)
        
if __name__ == "__main__":
    create_hashes()
