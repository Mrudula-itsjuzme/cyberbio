import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_proposals: int = 5, seed: Optional[int] = None) -> Dict[str, Any]:
        pass

class MockLLMProvider(LLMProvider):
    def __init__(self, mode: str = "valid"):
        self.mode = mode
        self.rng = np.random.default_rng(42)
        
    def generate(self, prompt: str, max_proposals: int = 5, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        if self.mode == "malformed":
            content = "This is not json"
        elif self.mode == "empty":
            content = json.dumps({"proposals": []})
        elif self.mode == "rate_limit":
            return {"error": "rate_limit", "calls": 1, "content": ""}
        else:
            content = json.dumps({
                "proposals": [
                    {
                        "sequence": f"[*]C([*])C{'C' * self.rng.integers(1, 10)}", 
                        "edit_description": "mocked", 
                        "rationale": "mock rationale"
                    } for _ in range(max_proposals)
                ]
            })
            
        return {
            "content": content,
            "input_tokens": 100,
            "output_tokens": 50 * max_proposals,
            "calls": 1
        }

class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        
        url_raw = base_url or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        # Ensure it's not a markdown link
        if url_raw.startswith("[") and "](" in url_raw:
            url_raw = url_raw.split("](")[1].strip(")")
        self.base_url = url_raw
        
        if not self.api_key:
            logging.warning("OPENAI_API_KEY is not set. Provider calls will fail.")

    def generate(self, prompt: str, max_proposals: int = 5, seed: Optional[int] = None) -> Dict[str, Any]:
        import urllib.request
        import urllib.error
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = (
            "You are an expert computational chemist. Output ONLY valid JSON matching this schema:\n"
            "{\n"
            "  \"proposals\": [\n"
            "    {\n"
            "      \"sequence\": \"<SMILES string>\",\n"
            "      \"edit_description\": \"<description>\",\n"
            "      \"rationale\": \"<reason>\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"}
        }
        if seed is not None:
            payload["seed"] = seed
            
        try:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {
                "content": content,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "calls": 1
            }
        except urllib.error.HTTPError as e:
            if e.code == 429:
                return {"content": "", "calls": 1, "error": "rate_limit"}
            return {"content": "", "calls": 1, "error": f"provider_error: {e.code} {e.reason}"}
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                return {"content": "", "calls": 1, "error": "timeout"}
            return {"content": "", "calls": 1, "error": f"provider_error: {str(e.reason)}"}
        except Exception as e:
            return {"content": "", "calls": 1, "error": f"provider_error: {str(e)}"}
