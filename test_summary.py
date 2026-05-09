import asyncio
import httpx
import re

# ==========================================
# HÀM GỌI LLM (TÓM TẮT 1 ĐOẠN TEXT)
# ==========================================
async def summarize_single_page(client: httpx.AsyncClient, page_num: str, text_content: str) -> tuple:
    """Gọi API Qwen2.5 để tóm tắt nội dung 1 trang."""
    
    # URL của SGLang server (Đệ nhớ kiểm tra lại port)
    end_point = "http://localhost:5001/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    model_name = "Qwen2.5/Qwen2.5-7B-Instruct/" 

    prompt = f"""Bạn là một chuyên gia phân tích tài liệu. Hãy đọc nội dung của Trang {page_num} sau đây (văn bản được trích xuất từ OCR) và thực hiện 2 nhiệm vụ:

1. TRÍCH XUẤT TỪ KHÓA: Liệt kê các thuật ngữ chuyên ngành, tên riêng, số liệu quan trọng hoặc khái niệm cốt lõi xuất hiện trong văn bản.
2. TÓM TẮT CHI TIẾT: Dựa trên các từ khóa vừa tìm được, hãy tóm tắt lại nội dung một cách logic, giữ nguyên các thuật ngữ quan trọng. Trình bày dưới dạng gạch đầu dòng để dễ đọc.

Nội dung Trang {page_num}:
{text_content}

Kết quả phân tích:"""

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
# HÀM XỬ LÝ CHÍNH: ĐỌC FILE -> GOM TRANG -> TÓM TẮT
# ==========================================
async def process_and_summarize(input_txt_path: str, output_txt_path: str):
    print(f"Đang đọc file OCR: {input_txt_path}")
    
    try:
        with open(input_txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_txt_path}!")
        return

    # 1. GOM TEXT THEO TỪNG TRANG
    # Dùng Regex để tìm các đoạn bắt đầu bằng "--- TRANG X | CỘT Y ---"
    # Logic: Chia nhỏ file text dựa trên chữ "--- TRANG"
    pages_data = {} # Format: {'1': "text cột 1 \n text cột 2", '2': "..."}
    
    # Cắt text thành từng khối dựa trên tiêu đề
    blocks = re.split(r'--- TRANG (\d+) \| CỘT \d+ ---', raw_text)
    
    # blocks[0] thường là rỗng hoặc rác đầu file.
    # Từ blocks[1] trở đi: blocks[i] là số trang, blocks[i+1] là nội dung
    for i in range(1, len(blocks), 2):
        page_num = blocks[i]
        content = blocks[i+1].strip()
        
        if page_num not in pages_data:
            pages_data[page_num] = ""
        pages_data[page_num] += content + "\n\n"

    print(f"Đã gom được {len(pages_data)} trang. Bắt đầu tóm tắt đồng thời...")

    # 2. GỌI LLM TÓM TẮT ĐỒNG THỜI (ASYNC)
    summaries = []
    async with httpx.AsyncClient() as client:
        # Tạo danh sách các task (mỗi task tóm tắt 1 trang)
        tasks = [
            summarize_single_page(client, p_num, p_content) 
            for p_num, p_content in pages_data.items()
        ]
        
        # Chạy tất cả các task cùng lúc
        results = await asyncio.gather(*tasks)

    # 3. LƯU KẾT QUẢ RA FILE
    # Sắp xếp lại kết quả theo số trang (vì chạy async nên kết quả trả về có thể lộn xộn)
    results.sort(key=lambda x: int(x[0]))
    
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for page_num, summary in results:
            f.write(f"=== TÓM TẮT TRANG {page_num} ===\n")
            f.write(f"{summary}\n\n")
            f.write("="*40 + "\n\n")

    print(f"\n========= HOÀN THÀNH =========")
    print(f"Đã lưu kết quả tóm tắt vào: {output_txt_path}")

# --- CHẠY THỬ ---
if __name__ == "__main__":
    INPUT_FILE = "ket_qua_ocr_final.txt"
    OUTPUT_FILE = "summary.txt"
    
    # Chạy hàm async trong môi trường đồng bộ
    asyncio.run(process_and_summarize(INPUT_FILE, OUTPUT_FILE))