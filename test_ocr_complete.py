
import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import requests
from paddleocr import LayoutDetection

# ==========================================
# 1. HÀM GỌI MISTRAL OCR (Dùng chung cho cả PDF và Ảnh)
# ==========================================
def call_mistral_ocr(file_bytes, filename):
    url = "http://192.168.20.156:8088/v1/ocr"
    data = {
        "model": "mistral-ocr-latest",
        "include_image_base64": "false" 
    }
    
    # Xác định mimetype dựa vào đuôi file
    mime_type = "application/pdf" if filename.lower().endswith('.pdf') else "image/png"
    
    files = {
        "file": (filename, file_bytes, mime_type)
    }
    
    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            result_json = response.json()
            extracted_page = result_json.get("pages", [{}])[0]
            extracted_text = extracted_page.get("markdown", "") 
            return extracted_text
        else:
            print(f"[Lỗi API] {filename}: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        print(f"[Lỗi Kết Nối] {filename}: {e}")
        return ""

# ==========================================
# 2. HÀM GOM NHÓM LOGIC (Tách ra cho gọn code)
# ==========================================
def group_layout_boxes(boxes, center_tolerance=50):
    text_boxes = [b for b in boxes if b['label'] == 'text']
    title_boxes = [b for b in boxes if b['label'] == 'paragraph_title']
    groups = []

    # BƯỚC 1: GOM CỤM (Sắp xếp theo chiều rộng giảm dần)
    text_boxes = sorted(text_boxes, key=lambda b: (b['coordinate'][2] - b['coordinate'][0]), reverse=True)
    for box in text_boxes:
        x_min, y_min, x_max, y_max = box['coordinate']
        center_x = (x_min + x_max) / 2
        placed = False
        for group in groups:
            cond1 = abs(center_x - group['center_x']) < center_tolerance
            cond2 = (x_min >= group['x_min'] - 20) and (x_max <= group['x_max'] + 20)
            if cond1 or cond2:
                group['boxes'].append(box)
                group['center_x'] = sum((b['coordinate'][0] + b['coordinate'][2])/2 for b in group['boxes']) / len(group['boxes'])
                group['x_min'] = min(group['x_min'], x_min)
                group['x_max'] = max(group['x_max'], x_max)
                group['y_min'] = min(group['y_min'], y_min)
                group['y_max'] = max(group['y_max'], y_max)
                placed = True
                break
        if not placed:
            groups.append({'center_x': center_x, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max, 'boxes': [box]})

    # BƯỚC 2: KÉO TITLE VÀO GROUP
    for title in title_boxes:
        tx_min, ty_min, tx_max, ty_max = title['coordinate']
        t_center_x = (tx_min + tx_max) / 2
        for group in groups:
            if (group['x_min'] - center_tolerance <= t_center_x <= group['x_max'] + center_tolerance) and (ty_min <= group['y_min'] + 50):
                group['boxes'].append(title)
                group['y_min'] = min(group['y_min'], ty_min)
                group['x_min'] = min(group['x_min'], tx_min)
                group['x_max'] = max(group['x_max'], tx_max)
                break

    # Sắp xếp các cột từ Trái sang Phải
    return sorted(groups, key=lambda g: g['x_min'])

# ==========================================
# 3. HÀM XỬ LÝ CHÍNH (Tự động nhận diện PDF / Ảnh)
# ==========================================
def process_and_ocr_document(file_path, center_tolerance=50):
    print(f"Đang phân tích layout cho: {file_path}")
    model = LayoutDetection(model_name="PP-DocLayout-L")
    
    is_pdf = file_path.lower().endswith('.pdf')
    all_extracted_text = []

    # Chuẩn bị dữ liệu lặp
    if is_pdf:
        doc = fitz.open(file_path)
        num_pages = len(doc)
    else:
        num_pages = 1
        img_orig = cv2.imread(file_path)
        if img_orig is None:
            print(f"Lỗi: Không thể đọc ảnh {file_path}")
            return

    for page_idx in range(num_pages):
        print(f"\n--- Đang xử lý trang {page_idx + 1}/{num_pages} ---")
        
        # --- A. LẤY ẢNH CV2 CHO PADDLE ---
        if is_pdf:
            page = doc[page_idx]
            page_rect = page.rect
            pix = page.get_pixmap()
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        else:
            img_cv2 = img_orig

        # --- B. CHẠY PADDLE & GOM NHÓM ---
        output = model.predict(img_cv2)
        res = output[0] if isinstance(output, list) else output
        
        boxes = []
        if isinstance(res, dict):
            if 'res' in res and 'boxes' in res['res']:
                boxes = res['res']['boxes']
            elif 'boxes' in res:
                boxes = res['boxes']
                
        if not boxes:
            print(f"Trang {page_idx + 1} trống.")
            continue

        groups = group_layout_boxes(boxes, center_tolerance)

        # --- C. CẮT VÀ GỬI LÊN MISTRAL OCR ---
        for col_idx, group in enumerate(groups):
            pad = 5
            
            if is_pdf:
                x1, y1 = group['x_min'] - pad, group['y_min'] - pad
                x2, y2 = group['x_max'] + pad, group['y_max'] + pad
                crop_rect = fitz.Rect(x1, y1, x2, y2) & page_rect
                
                if crop_rect.is_empty or not crop_rect.is_valid:
                    continue

                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                new_page = new_doc[0]
                new_page.set_cropbox(crop_rect)
                
                file_bytes = new_doc.tobytes()
                new_doc.close()
                filename_gia = f"page_{page_idx+1}_col_{col_idx+1}.pdf"
                
            else:
                x1 = max(0, int(group['x_min']) - pad)
                y1 = max(0, int(group['y_min']) - pad)
                x2 = min(img_cv2.shape[1], int(group['x_max']) + pad)
                y2 = min(img_cv2.shape[0], int(group['y_max']) + pad)
                
                cropped_img = img_cv2[y1:y2, x1:x2]
                # Mã hóa ảnh thành bytes trên RAM
                _, buffer = cv2.imencode('.png', cropped_img)
                file_bytes = buffer.tobytes()
                filename_gia = f"page_{page_idx+1}_col_{col_idx+1}.png"

            print(f"  -> Đang OCR {filename_gia}...")
            text_result = call_mistral_ocr(file_bytes, filename_gia)
            
            if text_result:
                all_extracted_text.append(f"--- TRANG {page_idx+1} | CỘT {col_idx+1} ---\n{text_result}\n")

    if is_pdf:
        doc.close()
    
    # --- D. LƯU KẾT QUẢ ---
    final_text = "\n".join(all_extracted_text)
    with open("ket_qua_ocr_final_hehe.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
        
    print("\n========= HOÀN THÀNH =========")
    print("Đã lưu toàn bộ chữ vào file: ket_qua_ocr_final.txt")

# --- CHẠY THỬ ---
if __name__ == "__main__":
    # Đệ có thể truyền vào file .pdf hoặc .png/.jpg đều chạy mượt!
    # file_path = "image.png" 
    file_path = "image.png" 
    process_and_ocr_document(file_path)