from dataclasses import dataclass, field
from config import load_config
from memory_store import (
    SYSTEM_PROMPT,
    estimate_tokens,
    extract_profile_updates,
    offline_response,
)
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    def __init__(self, config=None, force_offline=False):
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions = {}
        self.owners = {}
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
        return self._reply_offline(thread_id, message)

    def _reply_offline(self, thread_id, message):
        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({"role": "user", "content": message})
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + state.messages
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in prompt)
        facts = {}
        for item in state.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        if self.langchain_agent:
            result = self.langchain_agent.invoke(prompt)
            response = (
                result.content
                if isinstance(result.content, str)
                else str(result.content)
            )
            usage = result.usage_metadata or {}
        else:
            response, usage = offline_response(message, facts), {}
        output_tokens = usage.get("output_tokens", estimate_tokens(response))
        prompt_tokens = usage.get("input_tokens", prompt_tokens)
        state.token_usage += output_tokens
        state.prompt_tokens_processed += prompt_tokens
        state.messages.append({"role": "assistant", "content": response})
        return {
            "response": response,
            "agent_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "mode": "live" if self.langchain_agent else "offline",
        }

    def token_usage(self, thread_id):
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id):
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id):
        return 0
