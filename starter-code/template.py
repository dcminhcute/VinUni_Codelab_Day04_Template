import re
from typing import Any, Dict, List, Tuple

from tools import TOOL_DEFINITIONS, TOOL_MAP

# ---------------------------------------------------------------------------
# Milestone 1: Production System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast (xe điện), Vinpearl (du lịch),
  đồng thời hỗ trợ tiếp nhận và xử lý yêu cầu chăm sóc khách hàng (CSKH).
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, luôn trả lời bằng tiếng Việt.

## AVAILABLE TOOLS
1. search_product_catalog(category, max_price) — Tra cứu sản phẩm/dịch vụ Vingroup
   theo danh mục ('xe_dien' hoặc 'du_lich') và mức giá tối đa (VNĐ).
2. submit_support_ticket(customer_name, issue_description, priority) — Tạo ticket hỗ trợ
   khi khách hàng gặp vấn đề cần CSKH xử lý, trả về mã ticket để theo dõi.

## CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm, giá cả, hoặc trạng thái ticket. PHẢI gọi tool
   tương ứng để lấy dữ liệu thực trước khi trả lời.
2. Nếu câu hỏi vừa cần tra cứu sản phẩm vừa cần tạo ticket, PHẢI thực hiện CẢ HAI
   bước một cách độc lập, không được bỏ sót.
3. Nếu không tìm thấy sản phẩm phù hợp, PHẢI thông báo rõ ràng, không tự suy diễn
   sản phẩm khác.
4. KHÔNG cung cấp thông tin về đối thủ cạnh tranh hoặc các chủ đề ngoài phạm vi
   sản phẩm/dịch vụ Vingroup.

## OPERATIONAL BOUNDARIES
- Chỉ tư vấn về các sản phẩm và dịch vụ thuộc hệ sinh thái Vingroup (VinFast, Vinpearl,
  và các dịch vụ CSKH liên quan).
- Từ chối lịch sự các yêu cầu nằm ngoài phạm vi này (ví dụ: tư vấn tài chính cá nhân,
  y tế, pháp lý không liên quan đến sản phẩm Vingroup).

## OUTPUT CONTRACT
Khi suy luận nội bộ, tuân theo định dạng:
Thought: <suy nghĩ về việc cần làm gì tiếp theo>
Action: <tên tool cần gọi, nếu có>
Observation: <kết quả trả về từ tool>
...
Final Answer: <câu trả lời cuối cùng, rõ ràng, thân thiện, bằng tiếng Việt, dựa
hoàn toàn trên dữ liệu thực từ Observation>
"""


# ---------------------------------------------------------------------------
# Milestone 3 & 4: Agent Loop with Safeguards
# ---------------------------------------------------------------------------

class ToolCallingAgent:
    """
    Agent điều phối việc gọi tool dựa trên intent detection đơn giản
    (keyword matching), thực thi từng bước theo Agent Loop, và tổng hợp
    Final Answer từ các observation thu được.
    """

    CATALOG_KEYWORDS = [
        "xe điện", "vinfast", "giá", "mua", "sản phẩm", "du lịch",
        "vinpearl", "gói", "tour", "resort", "combo"
    ]
    TICKET_KEYWORDS = [
        "hỗ trợ", "khiếu nại", "lỗi", "hư", "bảo hành", "ticket",
        "phản ánh", "sự cố", "không hoạt động", "hoàn tiền"
    ]

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    # Intent Detection
    # -----------------------------------------------------------------
    def _detect_intents(self, user_input: str) -> Dict[str, bool]:
        text = user_input.lower()

        # Kiểm tra needs_catalog và needs_ticket ĐỘC LẬP (không dùng if-elif)
        # để tránh Trap 3: câu hỏi chứa cả hai loại nhu cầu cùng lúc.
        needs_catalog = any(kw in text for kw in self.CATALOG_KEYWORDS)
        needs_ticket = any(kw in text for kw in self.TICKET_KEYWORDS)
        is_faq = not needs_catalog and not needs_ticket

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq,
        }

    # -----------------------------------------------------------------
    # Helpers to pull structured params out of free-text input
    # -----------------------------------------------------------------
    @staticmethod
    def _extract_category(text: str) -> str:
        text = text.lower()
        if any(kw in text for kw in ["xe điện", "vinfast", "ô tô", "xe"]):
            return "xe_dien"
        if any(kw in text for kw in ["du lịch", "vinpearl", "tour", "resort"]):
            return "du_lich"
        return "xe_dien"

    @staticmethod
    def _extract_max_price(text: str) -> int:
        # Chỉ nhận số có kèm đơn vị "triệu"/"tỷ" ngay sau (ví dụ "800 triệu",
        # "1.2 tỷ"). Yêu cầu ký tự trước số không phải chữ/số để tránh bắt
        # nhầm số nằm trong tên model, ví dụ "VF8".
        match = re.search(
            r"(?<![a-z0-9])(\d+(?:[.,]\d+)?)\s*(tỷ|ty|triệu|trieu)\b",
            text.lower(),
        )
        if not match:
            return 999999999999
        value = float(match.group(1).replace(",", "."))
        unit = match.group(2)
        if unit in ("tỷ", "ty"):
            return int(value * 1_000_000_000)
        return int(value * 1_000_000)

    @staticmethod
    def _extract_priority(text: str) -> str:
        text = text.lower()
        if any(kw in text for kw in ["gấp", "khẩn cấp", "urgent", "nghiêm trọng"]):
            return "high"
        if any(kw in text for kw in ["không gấp", "khi nào rảnh", "low"]):
            return "low"
        return "medium"

    # -----------------------------------------------------------------
    # Single-step execution (one tool call or the final synthesis step)
    # -----------------------------------------------------------------
    def _execute_step(
        self, user_input: str, intents: Dict[str, bool], iteration: int
    ) -> Tuple[str, bool]:
        """
        Thực thi một bước trong Agent Loop.
        Trả về (result, is_final).
        """

        # Iteration 1: xử lý catalog nếu cần và chưa thực hiện
        if intents["needs_catalog"] and not self._has_step("search_product_catalog"):
            category = self._extract_category(user_input)
            max_price = self._extract_max_price(user_input)

            self.trace.append({
                "step": iteration,
                "thought": f"Người dùng cần thông tin sản phẩm loại '{category}'.",
                "action": "search_product_catalog",
                "action_input": {"category": category, "max_price": max_price},
            })

            try:
                results = TOOL_MAP["search_product_catalog"](category, max_price)
            except Exception as e:
                results = [{"error": str(e)}]

            self.trace[-1]["observation"] = results
            return "", False

        # Iteration 2: xử lý ticket nếu cần và chưa thực hiện
        if intents["needs_ticket"] and not self._has_step("submit_support_ticket"):
            customer_name = self._extract_customer_name(user_input)
            priority = self._extract_priority(user_input)

            self.trace.append({
                "step": iteration,
                "thought": "Người dùng cần tạo ticket hỗ trợ.",
                "action": "submit_support_ticket",
                "action_input": {
                    "customer_name": customer_name,
                    "issue_description": user_input,
                    "priority": priority,
                },
            })

            try:
                result = TOOL_MAP["submit_support_ticket"](
                    customer_name, user_input, priority
                )
            except Exception as e:
                result = {"error": str(e)}

            self.trace[-1]["observation"] = result
            return "", False

        # FAQ / không cần tool nào
        if intents["is_faq"]:
            answer = (
                "Cảm ơn bạn đã liên hệ VinAssistant! Bạn có thể hỏi mình về sản phẩm "
                "xe điện VinFast, các gói du lịch Vinpearl, hoặc gửi yêu cầu hỗ trợ "
                "nếu đang gặp vấn đề cần CSKH xử lý."
            )
            self.trace.append({
                "step": iteration,
                "thought": "Câu hỏi không cần tool, trả lời trực tiếp.",
                "action": None,
                "observation": None,
            })
            return answer, True

        # Tất cả các bước cần thiết đã hoàn tất -> Tổng hợp Final Answer
        final_answer = self._synthesize_final_answer(intents)
        return final_answer, True

    def _has_step(self, action_name: str) -> bool:
        return any(step.get("action") == action_name for step in self.trace)

    @staticmethod
    def _extract_customer_name(user_input: str) -> str:
        match = re.search(r"tên (?:tôi là|là|mình là)\s+([^\.,;\n]+)", user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Khách hàng chưa cung cấp tên"

    def _synthesize_final_answer(self, intents: Dict[str, bool]) -> str:
        parts = []

        if intents["needs_catalog"]:
            catalog_step = next(
                (s for s in self.trace if s.get("action") == "search_product_catalog"),
                None,
            )
            results = catalog_step["observation"] if catalog_step else []

            # Empty Results Handling (Milestone 4.2)
            if not results or (isinstance(results, list) and len(results) == 0):
                parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn.")
            elif isinstance(results, list) and results and "error" in results[0]:
                parts.append(f"Không thể tra cứu sản phẩm lúc này: {results[0]['error']}")
            else:
                lines = [
                    f"- {p.get('name', 'Sản phẩm')}: {p.get('price_vnd', 'N/A'):,} VNĐ"
                    if isinstance(p.get("price_vnd"), (int, float))
                    else f"- {p.get('name', 'Sản phẩm')}"
                    for p in results
                ]
                parts.append("Dưới đây là các sản phẩm phù hợp:\n" + "\n".join(lines))

        if intents["needs_ticket"]:
            ticket_step = next(
                (s for s in self.trace if s.get("action") == "submit_support_ticket"),
                None,
            )
            ticket_result = ticket_step["observation"] if ticket_step else {}
            if ticket_result.get("ticket_id"):
                parts.append(
                    f"Yêu cầu hỗ trợ của bạn đã được ghi nhận với mã ticket "
                    f"{ticket_result['ticket_id']}. Đội ngũ CSKH sẽ liên hệ sớm nhất."
                )
            else:
                parts.append("Rất tiếc, hiện chưa thể tạo ticket hỗ trợ. Vui lòng thử lại sau.")

        if not parts:
            parts.append("Xin lỗi, mình chưa xử lý được yêu cầu này. Bạn có thể nói rõ hơn không?")

        return "\n\n".join(parts)

    # -----------------------------------------------------------------
    # Public entrypoint: run the full Agent Loop
    # -----------------------------------------------------------------
    def run(self, user_input: str) -> Dict[str, Any]:
        self.trace = []
        intents = self._detect_intents(user_input)

        iteration = 1
        while iteration <= self.max_iterations:
            result, is_final = self._execute_step(user_input, intents, iteration)
            if is_final:
                return {"answer": result, "trace": self.trace, "status": "completed"}
            iteration += 1

        # Milestone 4.1: Max Iterations Guard
        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "status": "max_iterations_reached",
        }


if __name__ == "__main__":
    agent = ToolCallingAgent()
    demo_queries = [
        "Cho tôi xem xe điện VinFast dưới 800 triệu",
        "Tên tôi là Minh, xe VF8 tôi mua bị lỗi pin, cần hỗ trợ gấp",
        "Vingroup có những dịch vụ gì?",
    ]
    for q in demo_queries:
        print("USER:", q)
        print(agent.run(q))
        print("-" * 60)