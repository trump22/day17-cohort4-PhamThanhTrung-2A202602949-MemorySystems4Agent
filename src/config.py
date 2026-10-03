from dataclasses import dataclass
from pathlib import Path
import os
from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    offline: bool = True


def load_config(base_dir=None):
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    def model(prefix):
        p = normalize_provider(
            os.getenv(prefix + "_PROVIDER", os.getenv("LLM_PROVIDER", "openai"))
        )
        defaults = {
            "openai": "gpt-4o-mini",
            "custom": "local-model",
            "gemini": "gemini-2.5-flash",
            "anthropic": "claude-sonnet-4-5",
            "ollama": "llama3.2",
            "openrouter": "openai/gpt-4o-mini",
        }
        return ProviderConfig(
            p,
            os.getenv(prefix + "_MODEL", defaults[p]),
            float(os.getenv(prefix + "_TEMPERATURE", "0")),
            os.getenv(prefix + "_API_KEY") or os.getenv(p.upper() + "_API_KEY"),
            os.getenv(prefix + "_BASE_URL") or os.getenv(p.upper() + "_BASE_URL"),
        )

    threshold, keep = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200")), int(
        os.getenv("COMPACT_KEEP_MESSAGES", "4")
    )
    if min(threshold, keep) < 1:
        raise ValueError("Compact settings must be positive")
    state = Path(os.getenv("STATE_DIR", str(root / "state")))
    if not state.is_absolute():
        state = root / state
    state.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        root,
        root / "data",
        state,
        threshold,
        keep,
        model("LLM"),
        model("JUDGE"),
        os.getenv("LLM_MODE", "offline").lower() != "live",
    )
