# Báo cáo hoàn thành Day 17 — Memory Systems for AI Agent

## Phạm vi và cách chạy

Đã hoàn thiện toàn bộ scaffold trong `src/`: cấu hình, 6 provider, persistent profile, compact memory, hai agent, hai benchmark và 34 test. Dữ liệu gốc được giữ nguyên. Offline chạy không cần API key; live là phần mở rộng gọi chat model qua LangChain. Không cần LangGraph cho đường xử lý memory tự quản lý này.

```bash
python -m pip install -r requirements.txt
python -m pytest src/test_agents.py -v
python src/benchmark.py --output benchmark_results.json
```

## Kết quả đo thực tế

Kết quả dưới đây chạy offline, Python 3.14.4. Token heuristic bằng `ceil(len(text.strip()) / 4)`, không phải token billing của provider. Agent tokens only đếm token đầu ra; prompt tokens processed cộng ngữ cảnh đầu vào từng lần sinh câu trả lời, gồm system prompt. Cả lượt hội thoại và recall đều được tính. Các lần compact dùng quy tắc cục bộ nên không phát sinh token LLM bổ sung.

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 2615 | 23405 | 0.0% | 20.0% | 0 | 0 |
| Advanced | 2558 | 30629 | 100.0% | 100.0% | 260 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 449 | 24653 | 0.0% | 20.0% | 0 | 0 |
| Advanced | 460 | 14181 | 100.0% | 100.0% | 223 | 2 |

## Phân tích trade-off

Advanced đạt recall 100% trên hai dataset được giao vì hồ sơ `User.md` được dùng ở thread mới và vẫn tồn tại sau khi khởi tạo lại agent. Baseline đạt 0% cross-session recall vì chỉ đọc message trong thread hiện tại; test riêng xác nhận baseline vẫn nhớ đúng trong cùng thread.

Ở standard benchmark, Advanced xử lý thêm 30.9% prompt tokens. Hội thoại chưa đủ dài để compact kích hoạt, trong khi hồ sơ phải được đọc mỗi lượt. Vì vậy thêm persistent memory có chi phí kể cả khi câu trả lời không dài hơn.

Ở stress benchmark, Advanced compact 2 lần và giảm 42.5% prompt tokens. Compact giữ message gần đây, trích facts từ phần cũ và gộp vào summary trước đó. Summary chỉ giữ tối đa ba ghi chú, mỗi ghi chú 160 ký tự, nên không nối lịch sử vô hạn. Token đầu ra Advanced hơi cao hơn do trả lời được câu hỏi recall; đây là ví dụ tiết kiệm ngữ cảnh không đồng nghĩa giảm mọi loại token.

Memory growth đo chênh lệch bytes của hồ sơ trên đĩa, không bao gồm RAM hay summary. Profile standard kết thúc ở 260 bytes và stress ở 223 bytes. Upsert theo field thay thế giá trị cũ, bỏ ghi lại fact không thay đổi, nên lặp cùng một fact không làm file phình thêm. User riêng có file riêng; ID được sanitize kèm hash để tránh traversal và trùng tên.

## Bonus: structured extraction, conflict handling và bộ lọc độ chắc chắn

Fact được tách thành các field name, location, profession, response_style, interests, food, drink, pet. Chỉ trích từ phát biểu rõ về bản thân. Câu hỏi (kể cả không có dấu hỏi), giả định, nhắc lại và câu đùa bị loại khỏi đường ghi profile. Đây là bộ lọc quy tắc thiên về precision, không phải confidence score đã hiệu chuẩn.

Correction mới ghi đè field cũ: Huế/Đà Nẵng và backend/MLOps không được lưu song song trong profile. Bộ dữ liệu stress có Hà Nội đi họp và câu đùa product manager; chúng không thay đổi nơi ở/nghề nghiệp. Khi đọc summary cùng profile, profile hiện tại được ưu tiên. Test xác nhận correction, chống ghi nhầm câu hỏi, ghi lặp không tăng file và cách ly người dùng.

## Tính công bằng và tái lập

Hai agent dùng cùng extractor, cùng bộ sinh phản hồi offline và cùng estimator. Khác biệt nằm ở nguồn memory được cung cấp. Benchmark tạo thư mục state tạm mới cho mỗi suite và tự dọn sau khi chạy, tránh hồ sơ của lần trước làm tăng recall. Câu recall được hỏi ngay sau mỗi conversation trước correction tương lai, từng câu có thread riêng. Benchmark không đọc expected_contains để sinh câu trả lời; chúng chỉ dùng ở bước chấm.

Response quality là proxy offline: 80% mức phủ expected facts + 20% câu trả lời có nội dung và không quá 600 ký tự. Điểm 100% không chứng minh chất lượng hội thoại tổng quát hay tuân thủ mọi preference về trình bày. Vì vậy baseline vẫn có 20% dù recall bằng 0. JUDGE_PROVIDER/JUDGE_MODEL được cấu hình sẵn nhưng benchmark này không gọi LLM judge.

## Giới hạn và hướng phát triển

Regex phù hợp lab tiếng Việt nhưng có thể bỏ sót paraphrase hoặc nhận sai phát biểu phức tạp; bonus làm tăng precision và giảm recall ở cách diễn đạt chưa hỗ trợ. Summary có thể mất số liệu và chi tiết cũ; chưa triển khai semantic retrieval hay memory decay. Một message đơn lẻ rất dài có thể vượt ngưỡng vì hệ thống giữ nguyên message gần đây; ngưỡng compact là trigger, không phải hard context limit.

User.md chứa thông tin cá nhân dạng plaintext. Bản production cần quyền truy cập, thao tác xóa/sửa có kiểm soát, quản lý nhiều process ghi cùng file và chính sách giữ dữ liệu. Atomic replace giúp tránh file ghi dở nhưng chưa hỗ trợ transaction đa writer.

Live dùng cùng pipeline profile/compact và provider `.invoke()`. Tests kiểm tra prompt isolation, constructor arguments của 6 provider và token metadata bằng model giả. Chưa gọi API thật, chưa xác minh model availability hoặc latency/chi phí billing; không tự động chuyển offline khi live lỗi. OpenRouter đi qua API OpenAI-compatible. Tài liệu tích hợp: https://reference.langchain.com/python/integrations/overview và https://reference.langchain.com/python/langchain-openai/chat_models/base .
