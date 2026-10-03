from dataclasses import dataclass, asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import json
import unicodedata
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def recall_points(answer, expected):
    if not expected:
        return 1.0
    norm = lambda text: unicodedata.normalize("NFC", text).casefold()
    answer = norm(answer)
    return sum(norm(item) in answer for item in expected) / len(expected)


def heuristic_quality(answer, expected):
    """Offline proxy: factual coverage (80%) plus concise, nonempty response (20%).
    Not a human or LLM evaluation; cannot establish general response quality.
    """
    return 0.8 * recall_points(answer, expected) + 0.2 * bool(
        answer.strip() and len(answer) <= 600
    )


def run_agent_benchmark(agent_name, agent, conversations, config):
    users = {conv["user_id"] for conv in conversations}
    size = lambda: (
        sum(agent.memory_file_size(u) for u in users)
        if hasattr(agent, "memory_file_size")
        else 0
    )
    initial_size = size()
    scores, qualities, threads = [], [], set()
    for index, conv in enumerate(conversations):
        thread = f'{agent_name}:{index}:{conv["id"]}'
        threads.add(thread)
        for message in conv["turns"]:
            agent.reply(conv["user_id"], thread, message)
        # Evaluate immediately after each session, before future corrections.
        # Each recall question gets an independent thread with no prior messages.
        for number, item in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{thread}:recall:{number}"
            threads.add(recall_thread)
            answer = agent.reply(conv["user_id"], recall_thread, item["question"])[
                "response"
            ]
            scores.append(recall_points(answer, item["expected_contains"]))
            qualities.append(heuristic_quality(answer, item["expected_contains"]))
    return BenchmarkRow(
        agent_name,
        sum(agent.token_usage(t) for t in threads),
        sum(agent.prompt_token_usage(t) for t in threads),
        sum(scores) / len(scores) if scores else 0,
        sum(qualities) / len(qualities) if qualities else 0,
        size() - initial_size,
        sum(agent.compaction_count(t) for t in threads),
    )


def format_rows(rows):
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        values = [
            r.agent_name,
            str(r.agent_tokens_only),
            str(r.prompt_tokens_processed),
            f"{r.recall_score:.1%}",
            f"{r.response_quality:.1%}",
            str(r.memory_growth_bytes),
            str(r.compactions),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic offline memory benchmark; no API calls"
    )
    parser.add_argument("--output", type=Path, help="Write JSON metrics")
    args = parser.parse_args()
    config = load_config()
    results = {}
    for name, file in [
        ("Standard Benchmark", "conversations.json"),
        ("Long-Context Stress Benchmark", "advanced_long_context.json"),
    ]:
        conversations = load_conversations(config.data_dir / file)
        # Fresh state per suite/run prevents old profiles contaminating scores.
        with TemporaryDirectory(prefix="memory-benchmark-") as directory:
            isolated = replace(config, state_dir=Path(directory), offline=True)
            rows = [
                run_agent_benchmark(
                    "Baseline", BaselineAgent(isolated, True), conversations, isolated
                ),
                run_agent_benchmark(
                    "Advanced", AdvancedAgent(isolated, True), conversations, isolated
                ),
            ]
        print(name + "\n" + format_rows(rows) + "\n")
        results[name] = [asdict(row) for row in rows]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
