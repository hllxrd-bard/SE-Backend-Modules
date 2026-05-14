import asyncio
import httpx
import json

# ==========================================
# HÀM GỌI LLM (TÓM TẮT 1 ĐOẠN TEXT)
# ==========================================
async def summarize_single_page(client: httpx.AsyncClient, page_num: int, text_content: str) -> tuple:
    """Gọi API Qwen2.5 để tóm tắt nội dung 1 trang."""
    
    # URL của SGLang server (Đệ nhớ kiểm tra lại port)
    end_point = "http://localhost:5001/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    model_name = "Qwen/Qwen3.6-27B/" 

    prompt = f"""Bạn là một chuyên gia phân tích và tóm tắt tài liệu.

Hãy đọc kỹ nội dung của Trang {page_num} (được trích xuất từ OCR). Nhiệm vụ của bạn là tạo một bản tóm tắt có cấu trúc theo từng phần nội dung xuất hiện trong văn bản.

Yêu cầu:
- KHÔNG liệt kê từ khóa riêng lẻ.
- KHÔNG dùng markdown, bullet points, numbering, ký hiệu đặc biệt hoặc format trang trí.
- Chỉ trả về phần nội dung tóm tắt thuần văn bản.
- Tóm tắt phải bám sát nội dung gốc, theo đúng thứ tự xuất hiện trong tài liệu.
- Khi gặp thuật ngữ chuyên ngành, tên mô hình, công thức, số liệu, thuật toán, dataset, tên riêng hoặc khái niệm quan trọng, phải giữ nguyên chúng trong phần tóm tắt.
- Chia nội dung thành các đoạn ngắn tương ứng với từng ý/chủ đề của văn bản.
- Ưu tiên diễn giải logic giữa các ý thay vì liệt kê rời rạc.
- Không thêm kiến thức ngoài văn bản.
- Không suy diễn hoặc tự mở rộng nội dung không tồn tại trong OCR.
- Nếu văn bản OCR bị lỗi hoặc thiếu ngữ cảnh, hãy cố gắng suy luận tối thiểu dựa trên phần nhìn thấy được, nhưng không được bịa nội dung.

Nội dung Trang {page_num}:
{text_content}

Bản tóm tắt:"""

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1024
    }

    try:
        print(f"-> Đang gửi Trang {page_num} lên LLM...")
        # Timeout 120s vì LLM nghĩ hơi lâu
        response = await client.post(end_point, json=payload, headers=headers, timeout=120.0)
        if response.status_code == 200:
            summary = response.json()["choices"][0]["message"]["content"].strip()
            print(f"<- Đã nhận kết quả Trang {page_num}!")
            return (page_num, summary)
        else:
            err_msg = f"[Lỗi API: {response.status_code}]"
            print(f"<- Lỗi Trang {page_num}: {err_msg}")
            return (page_num, err_msg)
    except Exception as e:
        err_msg = f"[Lỗi Kết nối: {e}]"
        print(f"<- Lỗi Trang {page_num}: {err_msg}")
        return (page_num, err_msg)

# ==========================================
# HÀM XỬ LÝ CHÍNH: ĐỌC JSON -> GOM TRANG -> TÓM TẮT -> GHI SUMMARY VÀO JSON
# ==========================================
async def process_and_summarize(input_json_path: str, output_json_path: str):
    print(f"Đang đọc file OCR JSON: {input_json_path}")
    
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_json_path}!")
        return
    except json.JSONDecodeError as e:
        print(f"File JSON không hợp lệ: {e}")
        return

    # 1. GOM TEXT THEO TỪNG TRANG
    # Duyệt qua tất cả các block, gom ocr_text theo page number
    pages_data = {}  # Format: {1: "text block 1\ntext block 2\n...", 2: "..."}
    
    for block in ocr_data:
        page_num = block["page"]
        ocr_text = block.get("ocr_text", "").strip()
        
        if ocr_text:
            if page_num not in pages_data:
                pages_data[page_num] = ""
            pages_data[page_num] += ocr_text + "\n\n"

    print(f"Đã gom được {len(pages_data)} trang. Bắt đầu tóm tắt đồng thời...")

    # 2. GỌI LLM TÓM TẮT ĐỒNG THỜI (ASYNC)
    async with httpx.AsyncClient() as client:
        # Tạo danh sách các task (mỗi task tóm tắt 1 trang)
        tasks = [
            summarize_single_page(client, p_num, p_content) 
            for p_num, p_content in pages_data.items()
        ]
        
        # Chạy tất cả các task cùng lúc
        results = await asyncio.gather(*tasks)

    # 3. TẠO DICT: page_num -> summary
    summary_map = {page_num: summary for page_num, summary in results}

    # 4. GHI SUMMARY VÀO TỪNG BLOCK TRONG JSON
    for block in ocr_data:
        page_num = block["page"]
        block["summary"] = summary_map.get(page_num, "")

    # 5. LƯU KẾT QUẢ RA FILE JSON MỚI
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ocr_data, f, ensure_ascii=False, indent=2)

    print(f"\n========= HOÀN THÀNH =========")
    print(f"Đã lưu kết quả tóm tắt vào: {output_json_path}")

# --- CHẠY THỬ ---
if __name__ == "__main__":
    INPUT_FILE = "ket_qua_ocr_structured.json"
    OUTPUT_FILE = "ket_qua_ocr_structured_summarized.json"
    
    # Chạy hàm async trong môi trường đồng bộ
    asyncio.run(process_and_summarize(INPUT_FILE, OUTPUT_FILE))