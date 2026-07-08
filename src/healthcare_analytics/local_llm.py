"""Local open-source LLM wrapper.

The default runtime is Ollama, which lets the project behave like a normal
chatbot without an API key or Hugging Face LLM download. A Hugging Face provider
is still available as an optional fallback for users who prefer it.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .config import DEFAULT_LLM_CONTEXT_TOKENS, DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, DEFAULT_OLLAMA_BASE_URL


@dataclass
class LocalLLMConfig:
    model_name: str = DEFAULT_LLM_MODEL
    provider: str = DEFAULT_LLM_PROVIDER
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    max_new_tokens: int = DEFAULT_LLM_MAX_TOKENS
    context_tokens: int = DEFAULT_LLM_CONTEXT_TOKENS
    temperature: float = 0.2
    max_input_tokens: int = DEFAULT_LLM_CONTEXT_TOKENS


def clean_generated_text(text: str) -> str:
    """Normalize model text while preserving useful paragraph and bullet structure."""

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned_lines: list[str] = []
    blank_seen = False
    for line in lines:
        if not line:
            if not blank_seen and cleaned_lines:
                cleaned_lines.append("")
            blank_seen = True
            continue
        cleaned_lines.append(line)
        blank_seen = False
    return "\n".join(cleaned_lines).strip()


class LocalLLM:
    """Generate text with a local open-source model."""

    def __init__(self, config: LocalLLMConfig | None = None) -> None:
        self.config = config or LocalLLMConfig()
        self.provider = self.config.provider.lower().strip()
        self.model_name = self.config.model_name
        self._hf_loaded = False
        if self.provider == "ollama":
            try:
                import requests
            except ImportError as exc:
                raise RuntimeError(
                    "Ollama mode requires requests. Run: pip install -r requirements.txt"
                ) from exc
            self._requests = requests
            return

        self._load_huggingface_model()

    def _load_huggingface_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face LLM dependencies are not installed. Run: pip install transformers torch"
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model_kind = "seq2seq"
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
        except Exception:
            self.model_kind = "causal"
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self._hf_loaded = True

    def _generate_with_ollama(self, prompt: str, system_prompt: str | None = None) -> str:
        url = f"{self.config.base_url.rstrip('/')}/api/chat"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_new_tokens,
                "num_ctx": self.config.context_tokens,
            },
        }
        try:
            response = self._requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "Could not reach Ollama. Make sure the Ollama app is running and the model is pulled. "
                f"Try: ollama pull {self.config.model_name}"
            ) from exc

        data = response.json()
        message = data.get("message", {})
        text = message.get("content") or data.get("response") or ""
        return clean_generated_text(text)

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if self.provider == "ollama":
            return self._generate_with_ollama(prompt, system_prompt=system_prompt)

        if system_prompt:
            prompt = f"{system_prompt.strip()}\n\n{prompt.strip()}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens if self.config.max_new_tokens > 0 else 4096,
            "do_sample": self.config.temperature > 0,
            "temperature": self.config.temperature if self.config.temperature > 0 else None,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

        with self._torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)

        if self.model_kind == "causal":
            generated_ids = output[0][inputs["input_ids"].shape[-1] :]
        else:
            generated_ids = output[0]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return clean_generated_text(text)
