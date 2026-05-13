import asyncio
import httpx
import json
import re

# ==========================================
# HÀM GỌI LLM (TẠO FLASHCARD TỪ 1 BOX)
# ==========================================
async def create_flashcards_for_box(client: httpx.AsyncClient, group_idx: int, box_idx: int, text_content: str) -> tuple:
    """Gọi API Qwen2.5 để tạo 2 flashcard từ nội dung 1 box."""
    
    end_point = "http://localhost:5001/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    model_name = "Qwen/Qwen3.6-27B/"

    prompt = f"""You are a flashcard creator. Based on the following text content, create exactly 2 flashcards.

Each flashcard must have a QUESTION and an ANSWER. The question should test understanding of a key concept from the text. The answer should be concise and accurate.

You MUST use the following format exactly (do **NOT** add any extra text outside this format):

Do **NOT** put any inference or thinking process in the answer, just answer exactly 2 flashcards in the right format.

FLASHCARD_1
Q: <your question here>
A: <your answer here>

FLASHCARD_2
Q: <your question here>
A: <your answer here>

Text content (group {group_idx}, box {box_idx}):
{text_content}

Generate the 2 flashcards now:"""

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    try:
        print(f"-> Đang tạo flashcard cho Group {group_idx}, Box {box_idx}...")
        response = await client.post(end_point, json=payload, headers=headers, timeout=120.0)
        if response.status_code == 200:
            raw_output = response.json()["choices"][0]["message"]["content"].strip()
            print(f"<- Đã nhận kết quả Group {group_idx}, Box {box_idx}!")
            return (group_idx, box_idx, raw_output)
        else:
            err_msg = f"[Lỗi API: {response.status_code}]"
            print(f"<- Lỗi Group {group_idx}, Box {box_idx}: {err_msg}")
            return (group_idx, box_idx, err_msg)
    except Exception as e:
        err_msg = f"[Lỗi Kết nối: {e}]"
        print(f"<- Lỗi Group {group_idx}, Box {box_idx}: {err_msg}")
        return (group_idx, box_idx, err_msg)


# ==========================================
# HÀM PARSE FLASHCARD TỪ RAW OUTPUT CỦA LLM
# ==========================================
def parse_flashcards(raw_output: str) -> list:
    print(raw_output)
    """Parse raw LLM output thành list các flashcard dạng {question, answer}."""
    flashcards = []

    # Tìm tất cả các block FLASHCARD_N
    # Pattern: FLASHCARD_<số> theo sau bởi Q: ... và A: ...
    pattern = r"FLASHCARD_\d+\s*\n\s*Q:\s*(.+?)\s*\n\s*A:\s*(.+?)(?=\nFLASHCARD_|\Z)"
    matches = re.findall(pattern, raw_output, re.DOTALL)

    for question, answer in matches:
        flashcards.append({
            "question": question.strip(),
            "answer": answer.strip()
        })

    return flashcards


# ==========================================
# HÀM XỬ LÝ CHÍNH: ĐỌC JSON -> TẠO FLASHCARD THEO TỪNG BOX -> LƯU
# ==========================================
async def process_and_create_flashcards(input_json_path: str, output_json_path: str):
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

    # 1. DUYỆT TỪNG BOX ĐỂ TẠO FLASHCARD
    print(f"Tổng cộng {len(ocr_data)} box. Bắt đầu tạo flashcard...")

    async with httpx.AsyncClient() as client:
        tasks = []
        for block in ocr_data:
            group_idx = block["group_idx"]
            box_idx = block["box_idx"]
            ocr_text = block.get("ocr_text", "").strip()

            if ocr_text:
                tasks.append(
                    create_flashcards_for_box(client, group_idx, box_idx, ocr_text)
                )

        # Chạy tất cả các task cùng lúc
        results = await asyncio.gather(*tasks)

    # 2. PARSE VÀ GẮN FLASHCARD VÀO TỪNG BLOCK
    # Tạo map: (group_idx, box_idx) -> parsed flashcards
    flashcard_map = {}
    for group_idx, box_idx, raw_output in results:
        parsed = parse_flashcards(raw_output)
        flashcard_map[(group_idx, box_idx)] = parsed
        print(f"   Group {group_idx}, Box {box_idx}: Parsed {len(parsed)} flashcard(s)")

    # 3. GHI KEY "flashcards" VÀO TỪNG BLOCK TRONG JSON
    for block in ocr_data:
        key = (block["group_idx"], block["box_idx"])
        block["flashcards"] = flashcard_map.get(key, [])

    # 4. LƯU KẾT QUẢ RA FILE JSON MỚI
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ocr_data, f, ensure_ascii=False, indent=2)

    print(f"\n========= HOÀN THÀNH =========")
    print(f"Đã lưu kết quả flashcard vào: {output_json_path}")

    # 5. IN RA KẾT QUẢ ĐỂ KIỂM TRA
    total_cards = sum(len(block.get("flashcards", [])) for block in ocr_data)
    print(f"Tổng số flashcard đã tạo: {total_cards}")
    
    for block in ocr_data:
        cards = block.get("flashcards", [])
        if cards:
            print(f"\n--- Group {block['group_idx']}, Box {block['box_idx']} ---")
            for i, card in enumerate(cards, 1):
                print(f"  Flashcard {i}:")
                print(f"    Q: {card['question']}")
                print(f"    A: {card['answer']}")


# --- CHẠY THỬ ---
if __name__ == "__main__":
    INPUT_FILE = "ket_qua_ocr_structured.json"
    OUTPUT_FILE = "ket_qua_ocr_structured_flashcards.json"
    
    asyncio.run(process_and_create_flashcards(INPUT_FILE, OUTPUT_FILE))
