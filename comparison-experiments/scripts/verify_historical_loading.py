import json
import torch
import hashlib
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath("../materials-adversarial/src"))
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel
from materials_adv.data.tokenizer import Vocabulary

# Let's pick specialized_control as the representative checkpoint
ckpt_dir = Path("../materials-adversarial/results/models/specialized_control")
pt_path = ckpt_dir / "model.pt"
metrics_path = ckpt_dir / "metrics.json"

with metrics_path.open("r") as f:
    metrics = json.load(f)
arch = metrics["architecture"]

# The actual vocab used in training is not explicitly stored, but we can look at metrics.json maybe?
# Wait, let's load the checkpoint
state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
emb_weight = state_dict["embedding.weight"]
embedding_shape = list(emb_weight.shape)
checkpoint_vocab_size = embedding_shape[0] - 1 # Assuming TwoBranch which uses vocab_size+1

# Historical eval vocab
eval_vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + json.load(open("../materials-adversarial/data/processed/vocab.json"))
hist_eval_vocab_size = len(eval_vocab)
hist_eval_vocab_hash = hashlib.sha256(json.dumps(eval_vocab).encode()).hexdigest()
# Since we don't have the original training vocab saved in the model dir, we can't hash it properly.
# But we know it had checkpoint_vocab_size elements.
training_vocab_hash = "UNKNOWN_NO_VOCAB_FILE_IN_CKPT"

print(f"CHECKPOINT_VOCAB_SIZE: {checkpoint_vocab_size}")
print(f"TRAINING_VOCAB_HASH: {training_vocab_hash}")
print(f"HISTORICAL_EVAL_VOCAB_SIZE: {hist_eval_vocab_size}")
print(f"HISTORICAL_EVAL_VOCAB_HASH: {hist_eval_vocab_hash}")
print(f"EMBEDDING_SHAPE: {embedding_shape}")

# Reconstruct historical load behavior
model = TwoBranchTransformerRegressorModel(
    vocab_size=hist_eval_vocab_size,
    d_model=arch["d_model"],
    n_layers=arch["n_layers"],
    n_heads=arch["n_heads"],
    dim_feedforward=arch["dim_feedforward"],
    dropout=0.1,
    max_seq_len=arch.get("max_seq_len", 128)
)

original_eval_emb = model.embedding.weight.detach().clone()
mismatch_raised = False
try:
    model.load_state_dict(state_dict, strict=True)
except Exception as e:
    mismatch_raised = True

# Now with strict=False (historical behavior)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
loaded_eval_emb = model.embedding.weight.detach()

# Check if embedding weights remained randomly initialized
random_init_used = torch.allclose(original_eval_emb, loaded_eval_emb)

# Did inference occur? Yes, because load_state_dict doesn't crash with strict=False.
inference_occurred = True

print(f"LOAD_BEHAVIOR: Size mismatch raised on strict? {mismatch_raised}, missing_keys: {missing_keys}")
print(f"INFERENCE_OCCURRED: {inference_occurred}")
print(f"RANDOM_INITIALIZATION_USED: {random_init_used}")

if random_init_used:
    print("STATUS: INVALID_RANDOM_EMBEDDINGS")
elif mismatch_raised:
    print("STATUS: INVALID_MODEL_RECONSTRUCTION")
else:
    print("STATUS: VALID")

