from dataclasses import replace
from pathlib import Path
import json
import pytest
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import (
    UserProfileStore,
    CompactMemoryManager,
    extract_profile_updates,
    estimate_tokens,
)
from benchmark import load_conversations, run_agent_benchmark, recall_points
from model_provider import normalize_provider


def make_config(tmp_path):
    return replace(
        load_config(),
        state_dir=tmp_path,
        compact_threshold_tokens=220,
        compact_keep_messages=2,
        offline=True,
    )


def test_user_markdown_read_write_edit(tmp_path):
    store = UserProfileStore(tmp_path)
    assert store.read_text("a") == ""
    path = store.write_text("a", "# User\nDũngCT DũngCT")
    assert path.name == "User.md"
    assert store.file_size("a") == len(store.read_text("a").encode())
    assert store.edit_text("a", "DũngCT", "Trung")
    assert store.read_text("a") == "# User\nTrung DũngCT"
    assert not store.edit_text("a", "missing", "x")
    assert not store.edit_text("a", "", "x")


def test_compact_trigger(tmp_path):
    memory = CompactMemoryManager(100, 2)
    memory.append("t", "user", "Mình tên là Trung.")
    for i in range(8):
        memory.append("t", "user", f"Ghi chú {i}: " + "nội dung dài " * 150)
    state = memory.context("t")
    assert state["compactions"] >= 2
    assert len(state["messages"]) <= 2
    assert json.loads(state["summary"])["facts"]["name"] == "Trung"
    assert len(json.loads(state["summary"])["notes"]) <= 3
    assert memory.compaction_count("other") == 0


def test_cross_session_recall(tmp_path):
    config = make_config(tmp_path)
    a, b = AdvancedAgent(config, True), BaselineAgent(config, True)
    for agent in (a, b):
        agent.reply(
            "u",
            "old",
            "Mình tên là Trung. Mình ở Huế. Đồ uống yêu thích là cà phê sữa đá.",
        )
        assert "Trung" in agent.reply("u", "old", "Mình tên gì?")["response"]
    question = "Mình tên gì và đồ uống yêu thích là gì?"
    assert "Trung" not in b.reply("u", "new", question)["response"]
    assert "Trung" in a.reply("u", "new", question)["response"]
    restarted = AdvancedAgent(config, True)
    assert "cà phê sữa đá" in restarted.reply("u", "restart", question)["response"]
    assert "Trung" not in restarted.reply("other", "other-thread", question)["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path):
    config = make_config(tmp_path)
    a, b = AdvancedAgent(config, True), BaselineAgent(config, True)
    for i in range(20):
        message = f"Turn {i}: " + "Thông tin tạm trong hội thoại dài. " * 150
        for agent in (a, b):
            agent.reply("u", "t", message)
    assert a.compaction_count("t") > 1
    assert a.prompt_token_usage("t") < b.prompt_token_usage("t") * 0.5


@pytest.mark.parametrize(
    "message",
    [
        "Bạn có biết DũngCT không?",
        "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.",
        "Mình tên gì?",
        "Nếu mình ở Hà Nội thì sao?",
        "Mình không còn làm backend engineer nữa.",
        "Mình đùa rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa.",
        "Hà Nội chỉ là nơi mình vừa đi họp.",
        "Đừng nói backend engineer nữa vì đó là thông tin cũ.",
    ],
)
def test_noise_is_not_a_profile_fact(message):
    assert extract_profile_updates(message) == {}


def test_latest_correction_and_question_do_not_overwrite(tmp_path):
    a = AdvancedAgent(make_config(tmp_path), True)
    for message in [
        "Mình ở Đà Nẵng và đang làm backend engineer.",
        "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng.",
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
        "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.",
    ]:
        a.reply("u", "t", message)
    facts = a.profile_store.facts("u")
    assert facts["location"] == "Huế"
    assert facts["profession"] == "MLOps engineer"
    assert "drink" not in facts
    profile = a.profile_store.read_text("u")
    assert "backend engineer" not in profile and "Đà Nẵng" not in profile


def test_profiles_safe_and_no_slug_collision(tmp_path):
    store = UserProfileStore(tmp_path)
    for user in ["../../escape", "a/b", "a_b", "CON"]:
        assert store.path_for(user).resolve().is_relative_to(tmp_path.resolve())
    assert store.path_for("a/b") != store.path_for("a_b")
    with pytest.raises(ValueError):
        store.path_for("")


def test_repeat_fact_does_not_grow_memory(tmp_path):
    store = UserProfileStore(tmp_path)
    store.upsert_fact("u", "name", "Trung")
    before = store.read_text("u")
    for _ in range(20):
        store.upsert_fact("u", "name", "Trung")
    assert store.read_text("u") == before


@pytest.mark.parametrize("agent_class", [AdvancedAgent, BaselineAgent])
def test_thread_isolation_and_accounting(tmp_path, agent_class):
    agent = agent_class(make_config(tmp_path), True)
    results = [
        agent.reply("a", "t", "Mình tên là Trung."),
        agent.reply("a", "t", "Mình tên gì?"),
    ]
    assert agent.token_usage("t") == sum(r["agent_tokens"] for r in results)
    assert agent.prompt_token_usage("t") == sum(r["prompt_tokens"] for r in results)
    with pytest.raises(ValueError):
        agent.reply("b", "t", "Mình tên gì?")


def test_benchmark_on_supplied_datasets(tmp_path):
    config = replace(
        make_config(tmp_path), compact_threshold_tokens=1200, compact_keep_messages=4
    )
    for filename in ("conversations.json", "advanced_long_context.json"):
        conversations = load_conversations(config.data_dir / filename)
        advanced = run_agent_benchmark(
            "Advanced",
            AdvancedAgent(replace(config, state_dir=tmp_path / filename), True),
            conversations,
            config,
        )
        baseline = run_agent_benchmark(
            "Baseline", BaselineAgent(config, True), conversations, config
        )
        assert advanced.recall_score == 1
        assert baseline.recall_score == 0
        if "long" in filename:
            assert advanced.compactions > 1
            assert advanced.prompt_tokens_processed < baseline.prompt_tokens_processed


def test_token_and_recall_estimates():
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("12345") == 2
    assert recall_points("HUẾ", ["Huế", "Python"]) == 0.5
    assert recall_points("", ["a"]) == 0


@pytest.mark.parametrize(
    "provider", ["openai", "custom", "gemini", "anthropic", "ollama", "openrouter"]
)
def test_supported_providers(provider):
    assert normalize_provider(provider.upper()) == provider


def test_provider_validation():
    assert normalize_provider("anthorpic") == "anthropic"
    with pytest.raises(ValueError):
        normalize_provider("unknown")


@pytest.mark.parametrize(
    "provider,module,class_name",
    [
        ("openai", "langchain_openai", "ChatOpenAI"),
        ("custom", "langchain_openai", "ChatOpenAI"),
        ("openrouter", "langchain_openai", "ChatOpenAI"),
        ("gemini", "langchain_google_genai", "ChatGoogleGenerativeAI"),
        ("anthropic", "langchain_anthropic", "ChatAnthropic"),
        ("ollama", "langchain_ollama", "ChatOllama"),
    ],
)
def test_provider_constructor_arguments(monkeypatch, provider, module, class_name):
    import sys
    from types import SimpleNamespace
    from model_provider import ProviderConfig, build_chat_model

    monkeypatch.setitem(
        sys.modules, module, SimpleNamespace(**{class_name: lambda **kw: kw})
    )
    config = ProviderConfig(
        provider,
        "test-model",
        0,
        "test-key" if provider != "ollama" else None,
        "https://example.test/api",
    )
    result = build_chat_model(config)
    assert result["model"] == "test-model"
    assert result["temperature"] == 0
    assert result["base_url"] == "https://example.test/api"
    if provider != "ollama":
        assert result["api_key"] == "test-key"


@pytest.mark.parametrize("agent_class", [AdvancedAgent, BaselineAgent])
def test_live_path_uses_memory_and_real_usage(tmp_path, monkeypatch, agent_class):
    from types import SimpleNamespace
    import agent_advanced, agent_baseline

    prompts = []

    class FakeModel:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(
                content="Phản hồi live",
                usage_metadata={"input_tokens": 88, "output_tokens": 12},
            )

    for module in (agent_advanced, agent_baseline):
        monkeypatch.setattr(module, "build_chat_model", lambda config: FakeModel())
    agent = agent_class(replace(make_config(tmp_path), offline=False))
    agent.reply("u", "old", "Mình tên là Trung.")
    agent.reply("u", "new", "Mình tên gì?")
    prompt = str(prompts[-1])
    assert ("Trung" in prompt) == (agent_class is AdvancedAgent)
    assert agent.token_usage("new") == 12
    assert agent.prompt_token_usage("new") == 88
