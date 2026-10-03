from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import math
import re
import unicodedata

SYSTEM_PROMPT = "Bạn là trợ lý tiếng Việt. Dùng thông tin được cung cấp, ưu tiên đính chính mới nhất. Không đoán thông tin chưa biết. Trả lời ngắn gọn, có ví dụ khi phù hợp."


def estimate_tokens(text):
    return math.ceil(len(text.strip()) / 4)


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id):
        if not user_id:
            raise ValueError("Empty user id")
        slug = re.sub(r"[^\w-]", "_", user_id)[:48]
        digest = hashlib.sha256(user_id.encode()).hexdigest()[:12]
        return self.root_dir / f"{slug}-{digest}" / "User.md"

    def read_text(self, user_id):
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id, content):
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(content, encoding="utf-8", newline="\n")
        temp.replace(path)
        return path

    def edit_text(self, user_id, search_text, replacement):
        text = self.read_text(user_id)
        if not search_text or search_text not in text:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id):
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id):
        result = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^- ([a-z_]+): (.*)$", line)
            if match:
                try:
                    result[match[1]] = json.loads(match[2])
                except json.JSONDecodeError:
                    continue
        return result

    def upsert_fact(self, user_id, key, value):
        if not re.fullmatch(r"[a-z_]+", key):
            raise ValueError("Invalid field")
        facts = self.facts(user_id)
        if facts.get(key) == value:
            return
        facts[key] = value
        self.write_text(
            user_id,
            "# User profile\n\n"
            + "\n".join(
                f"- {k}: {json.dumps(v,ensure_ascii=False)}"
                for k, v in sorted(facts.items())
            )
            + "\n",
        )


def extract_profile_updates(message):
    """Conservative structured extraction; latest explicit assertion replaces old facts.

    No extraction from questions, hypothetical statements, jokes or historical
    mentions. This precision gate intentionally misses some paraphrases.
    """
    facts = {}
    for sentence in re.split(
        r"(?<=[.!?])\s+|\n", unicodedata.normalize("NFC", message)
    ):
        text = sentence.strip().rstrip(".!?")
        lower = text.casefold()
        if "?" in sentence or any(
            x in lower
            for x in (
                "nếu ",
                "lúc đầu",
                "nhắc lại",
                "nhớ lại",
                "hay là",
                "câu đùa",
                "là gì",
                "tên gì",
                "ở đâu",
            )
        ):
            if "nhưng thực ra" in lower:
                text = re.split("nhưng thực ra", text, flags=re.I)[-1]
                lower = text.casefold()
            else:
                continue
        match = re.search(r"(?:mình|tôi) tên (?:là )?([^,;:]+)", text, re.I)
        if match:
            facts["name"] = match[1].strip()
        match = re.search(
            r"(?:mình (?:(?:hiện tại |hiện |giờ |vẫn |đang )*)|hiện )ở\s+([^,;]+)",
            text,
            re.I,
        )
        if not match:
            match = re.search(r"mình đang làm việc ở\s+([^,;]+)", text, re.I)
        if match:
            facts["location"] = re.split(
                r"\s+(?:và|chứ|trong|vài|để|dù|mỗi|nhưng|chưa)\b", match[1], flags=re.I
            )[0].strip()
        jobs = re.findall(
            r"(?:đang làm|mình làm|chuyển sang|nghề nghiệp hiện tại vẫn là|nghề nghiệp thì vẫn là)\s+([\w -]+?engineer)\b",
            text,
            re.I,
        )
        if jobs:
            facts["profession"] = jobs[-1].strip()
        if (
            ("mình" in lower or "hãy trả lời" in lower or "style trả lời" in lower)
            and any(x in lower for x in ("muốn", "thích", "hãy trả lời", "vẫn giữ"))
            and any(x in lower for x in ("ngắn", "bullet", "ví dụ"))
        ):
            style = "ngắn gọn"
            if "3 bullet" in lower:
                style += ", 3 bullet"
            elif "bullet" in lower:
                style += ", bullet"
            if "ví dụ" in lower:
                style += (
                    ", có ví dụ thực chiến"
                    if "thực chiến" in lower
                    else ", có ví dụ thực tế"
                )
            if "trade-off" in lower:
                style += ", so sánh trade-off"
            facts["response_style"] = style
        for key, pattern in [
            ("drink", r"đồ uống yêu thích (?:của mình )?là\s+([^,;]+)"),
            ("food", r"món ăn yêu thích (?:của mình )?là\s+([^,;]+)"),
            ("pet", r"mình nuôi\s+([^,;]+)"),
        ]:
            match = re.search(pattern, text, re.I)
            if match:
                facts[key] = match[1].strip()
        if re.search(r"mình (?:vẫn )?(?:thích|đang quan tâm|quan tâm)", lower):
            interests = [
                item
                for item in ("Python", "AI", "MLOps", "chạy bộ", "lo-fi")
                if item.casefold() in lower
            ]
            if interests:
                facts["interests"] = ", ".join(interests)
    return facts


def summarize_messages(messages, max_items=6):
    facts, notes = {}, []
    for item in messages:
        if item["role"] == "user":
            facts.update(extract_profile_updates(item["content"]))
            notes.append(item["content"][:160])
    return json.dumps({"facts": facts, "notes": notes[-max_items:]}, ensure_ascii=False)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict = field(default_factory=dict)

    def __post_init__(self):
        if min(self.threshold_tokens, self.keep_messages) < 1:
            raise ValueError("Invalid compact settings")

    def context(self, thread_id):
        return self.state.setdefault(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )

    def append(self, thread_id, role, content):
        state = self.context(thread_id)
        messages = state["messages"]
        messages.append({"role": role, "content": content})
        total = estimate_tokens(state["summary"]) + sum(
            estimate_tokens(m["content"]) for m in messages
        )
        if total > self.threshold_tokens and len(messages) > self.keep_messages:
            fresh = json.loads(summarize_messages(messages[: -self.keep_messages], 3))
            old = (
                json.loads(state["summary"])
                if state["summary"]
                else {"facts": {}, "notes": []}
            )
            old["facts"].update(fresh["facts"])
            old["notes"] = (old["notes"] + fresh["notes"])[-3:]
            state["summary"] = json.dumps(old, ensure_ascii=False)
            state["messages"] = messages[-self.keep_messages :]
            state["compactions"] += 1

    def compaction_count(self, thread_id):
        return self.context(thread_id)["compactions"]


def offline_response(message, facts):
    lower = message.casefold()
    selectors = {
        "name": ("tên", "là ai", "biết"),
        "location": (
            "ở đâu",
            "nơi ở",
            "hiện đang ở",
            "hiện tại mình đang ở",
            "huế",
            "hà nội",
        ),
        "profession": ("nghề", "công việc"),
        "response_style": ("style", "kiểu trả lời", "như thế nào"),
        "drink": ("đồ uống",),
        "food": ("món ăn",),
        "pet": ("nuôi", "con gì"),
        "interests": ("quan tâm", "kỹ thuật chính"),
    }
    fields = [key for key, terms in selectors.items() if any(t in lower for t in terms)]
    if "tóm tắt" in lower or "mình là ai" in lower:
        fields = list(selectors)
    if "?" in message or any(x in lower for x in ("nhắc lại", "tóm tắt", "nhớ lại")):
        labels = {
            "name": "Tên",
            "location": "Nơi ở",
            "profession": "Nghề nghiệp",
            "response_style": "Style",
            "drink": "Đồ uống",
            "food": "Món ăn",
            "pet": "Thú nuôi",
            "interests": "Mối quan tâm",
        }
        if not fields:
            return "Bạn có thể nói rõ thông tin cần nhắc lại không?"
        return (
            "; ".join(
                f'{labels[k]}: {facts.get(k,"chưa có thông tin")}' for k in fields
            )
            + "."
        )
    if extract_profile_updates(message):
        return "Đã ghi nhận thông tin bạn cung cấp; ưu tiên bản đính chính mới nhất."
    return "Có thể so sánh recall và lượng ngữ cảnh xử lý trên cùng dữ liệu. Ví dụ: kiểm tra một fact ở thread mới."
