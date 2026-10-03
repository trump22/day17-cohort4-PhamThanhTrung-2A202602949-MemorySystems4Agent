from dataclasses import dataclass
import json
from config import load_config
from memory_store import (
    SYSTEM_PROMPT,
    UserProfileStore,
    CompactMemoryManager,
    estimate_tokens,
    extract_profile_updates,
    offline_response,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    def __init__(self, config=None, force_offline=False):
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            self.config.compact_threshold_tokens, self.config.compact_keep_messages
        )
        self.thread_tokens, self.thread_prompt_tokens, self.owners = {}, {}, {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def _maybe_build_langchain_agent(self):
        return (
            None
            if self.force_offline or self.config.offline
            else build_chat_model(self.config.model)
        )

    def reply(self, user_id, thread_id, message):
        owner = self.owners.setdefault(thread_id, user_id)
        if owner != user_id:
            raise ValueError("Thread belongs to another user")
        return self._reply_offline(user_id, thread_id, message)

    def _prompt(self, user_id, thread_id):
        state = self.compact_memory.context(thread_id)
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}]
        profile = self.profile_store.read_text(user_id)
        if profile:
            prompt.append(
                {
                    "role": "system",
                    "content": "Hồ sơ người dùng (ưu tiên fact hiện tại):\n" + profile,
                }
            )
        if state["summary"]:
            prompt.append(
                {
                    "role": "system",
                    "content": "Tóm tắt lịch sử (có thể chứa thông tin cũ):\n"
                    + state["summary"],
                }
            )
        return prompt + state["messages"]

    def _estimate_prompt_context_tokens(self, user_id, thread_id):
        return sum(
            estimate_tokens(item["content"])
            for item in self._prompt(user_id, thread_id)
        )

    def _offline_response(self, user_id, thread_id, message):
        state = self.compact_memory.context(thread_id)
        facts = json.loads(state["summary"])["facts"] if state["summary"] else {}
        for item in state["messages"]:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        facts.update(self.profile_store.facts(user_id))
        return offline_response(message, facts)

    def _reply_offline(self, user_id, thread_id, message):
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        if self.langchain_agent:
            result = self.langchain_agent.invoke(self._prompt(user_id, thread_id))
            response = (
                result.content
                if isinstance(result.content, str)
                else str(result.content)
            )
            usage = result.usage_metadata or {}
        else:
            response, usage = self._offline_response(user_id, thread_id, message), {}
        output_tokens = usage.get("output_tokens", estimate_tokens(response))
        prompt_tokens = usage.get("input_tokens", prompt_tokens)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + output_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.prompt_token_usage(thread_id) + prompt_tokens
        )
        self.compact_memory.append(thread_id, "assistant", response)
        return {
            "response": response,
            "agent_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "mode": "live" if self.langchain_agent else "offline",
        }

    def token_usage(self, thread_id):
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id):
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id):
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id):
        return self.compact_memory.compaction_count(thread_id)
