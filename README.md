Online Learning for LangGraph Agents
📖 Giới thiệu
Dự án này trình bày một hướng tiếp cận mới trong việc xây dựng các agent thông minh bằng cách kết hợp Học trực tuyến (Online Learning) vào kiến trúc đồ thị của LangGraph. Thay vì phụ thuộc vào việc fine-tuning LLM vốn tốn kém và có thể làm giảm tính tổng quát của mô hình, chúng tôi tập trung vào việc cải thiện khả năng tự điều chỉnh và thích nghi của agent trong quá trình thực thi.

Mục tiêu chính là tạo ra một agent có khả năng tự phục hồi và điều chỉnh chiến lược một cách thông minh, đặc biệt trong các tác vụ tương tác với công cụ (tool-use) như truy vấn cơ sở dữ liệu.

✨ Điểm nổi bật
create_react_agent_with_adaptive_planner: Một phiên bản mở rộng của langgraph.prebuilt.create_react_agent, tích hợp thuật toán Adaptive Planner dựa trên các phương pháp Học tăng cường trực tuyến (Online Reinforcement Learning - RL) và thuật toán PPO (Proximal Policy Optimization).

Vòng lặp tự phục hồi: Kiến trúc vòng lặp ngoài đồ thị cho phép agent xử lý lỗi, phân tích kết quả và điều chỉnh kế hoạch một cách thông minh.

Xử lý lỗi mạnh mẽ: Agent có khả năng thử lại với các tham số đã được điều chỉnh, tránh các vòng lặp thất bại và kết thúc quy trình một cách có kiểm soát khi hết số lần thử.

⚙️ Kiến trúc
Kiến trúc của agent được thiết kế như một đồ thị trạng thái (StateGraph) với một vòng lặp ngoài, cho phép agent có khả năng tự sửa lỗi và điều chỉnh chiến lược.

initial_retrieval: Bắt đầu bằng việc thu thập thông tin ban đầu từ cơ sở dữ liệu vector hoặc các nguồn khác.

planning_agent: "Bộ não" của agent. Dựa trên thông tin đầu vào và lịch sử, nó sẽ quyết định hành động tiếp theo, thường là tạo một truy vấn SQL.

run_query: Thực thi truy vấn SQL.

Thành công: Chuyển đến update_retry_count (để reset số lần thử lại) và sau đó quay lại planning_agent để diễn giải kết quả và đưa ra câu trả lời cuối cùng.

Thất bại: Chuyển đến error_analysis_node.

error_analysis_node: Dùng LLM để phân tích lỗi, tăng số lần thử lại và tạo ra một gợi ý để planning_agent có thể sửa chữa truy vấn.

final_failure: Nếu agent thất bại sau max_retries lần thử, quy trình sẽ kết thúc tại node này.

✨ Bonus
cách sửa thư viện sqlalchemy và langgraph để kết nối với hive  lakehouse hỗ trợ BI, DA, DE.
