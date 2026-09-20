with open("materials-adversarial/scripts/run_multiseed.py", "r") as f:
    content = f.read()

content = content.replace("from materials_adv.attacks.surrogate_utils import ModelAdapter", "")

adapter_code = """
class ModelAdapter:
    def __init__(self, model, vocab_map, scaler, max_len=128, device="cpu"):
        self.model = model
        self.vocab_map = vocab_map
        self.scaler = scaler
        self.max_len = max_len
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, reps: list[str]) -> list[float]:
        preds = []
        with torch.no_grad():
            for rep in reps:
                toks = tokenize(rep)
                slen = min(len(toks), self.max_len)
                ids = torch.zeros((1, self.max_len), dtype=torch.long, device=self.device)
                mask = torch.ones((1, self.max_len), dtype=torch.bool, device=self.device)
                for j in range(slen):
                    ids[0, j] = self.vocab_map.get(toks[j], 0)
                    mask[0, j] = False
                # Fix padding_mask vs mask call
                try:
                    spred = self.model(ids, padding_mask=mask).cpu().numpy().item()
                except:
                    spred = self.model(ids, mask).squeeze(-1).cpu().numpy().item()
                unscaled = float(self.scaler.inverse_transform(np.array([[spred]]))[0, 0])
                preds.append(unscaled)
        return preds
"""

# Insert right after SimpleDataset
idx = content.find("df = pd.read_csv")
content = content[:idx] + adapter_code + "\n" + content[idx:]

with open("materials-adversarial/scripts/run_multiseed.py", "w") as f:
    f.write(content)
