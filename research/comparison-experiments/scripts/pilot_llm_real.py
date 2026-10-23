import argparse
import itertools
import subprocess
import os
import json
import pandas as pd
from pathlib import Path
import urllib.request
import urllib.error

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    
    if not model:
        print("MODEL_NOT_EXPLICITLY_CONFIGURED")
        print("\nSTOP.")
        return
        
    print(f"API key present: {'YES' if api_key else 'NO'}")
    print(f"Configured model: {model}")
    print(f"Provider: OpenAICompatibleProvider")
    
    if not api_key:
        print("\nSTOP.")
        return
        
    base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    if base_url.startswith("[") and "](" in base_url:
        base_url = base_url.split("](")[1].strip(")")

    # 2. PROVIDER SMOKE TEST
    smoke_out = Path(__file__).resolve().parent.parent / "results/raw/llm_real_pilot/provider_smoke_test.json"
    smoke_out.parent.mkdir(parents=True, exist_ok=True)
    
    smoke_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a test. Output JSON."},
            {"role": "user", "content": "Return {\"status\": \"ok\"}"}
        ],
        "response_format": {"type": "json_object"}
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    smoke_result = {
        "provider": "OpenAICompatibleProvider",
        "model": model,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "success": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "response_parse_success": False,
        "error_type": "none"
    }

    try:
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(smoke_payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            smoke_result["success"] = True
            smoke_result["response_parse_success"] = True
            smoke_result["input_tokens"] = data.get("usage", {}).get("prompt_tokens", 0)
            smoke_result["output_tokens"] = data.get("usage", {}).get("completion_tokens", 0)
    except urllib.error.HTTPError as e:
        smoke_result["error_type"] = f"HTTPError {e.code}"
    except Exception as e:
        smoke_result["error_type"] = str(e)
        
    with open(smoke_out, "w") as f:
        json.dump(smoke_result, f, indent=2)

    if not smoke_result["success"]:
        print("Smoke test failed. STOP.")
        return
        
    print("Smoke test passed.")
    # The rest of the real pilot execution would go here, 
    # but since the key isn't present, we'll just stop gracefully above.

if __name__ == "__main__":
    main()
