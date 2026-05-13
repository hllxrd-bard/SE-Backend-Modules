
import os
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
import requests
from paddleocr import LayoutDetection

# ==========================================
# BẢNG MÀU CHO VISUALIZATION (BGR format cho OpenCV)
# ==========================================
LABEL_COLORS = {
    'text':             (255, 0, 0),       # Xanh dương
    'paragraph_title':  (0, 0, 255),       # Đỏ
    'table':            (0, 255, 0),       # Xanh lá
    'table_title':      (0, 200, 200),     # Vàng đậm
    'figure':           (255, 0, 255),     # Tím
    'figure_title':     (255, 100, 200),   # Hồng
    'formula':          (0, 165, 255),     # Cam
    'list':             (255, 255, 0),     # Cyan
    'abstract':         (0, 255, 255),     # Vàng
    'reference':        (128, 0, 128),     # Tím đậm
    'header':           (200, 200, 200),   # Xám nhạt
    'footer':           (150, 150, 150),   # Xám
}

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
# 2. HÀM VISUALIZE LAYOUT DETECTION
# ==========================================
def visualize_layout(img_cv2, boxes, page_idx, save_dir="layout_viz"):
    """
    Vẽ tất cả các bounding box lên ảnh với màu và label tương ứng.
    Lưu ảnh ra file.
    """
    os.makedirs(save_dir, exist_ok=True)
    img_viz = img_cv2.copy()
    
    for box in boxes:
        label = box['label']
        score = box.get('score', 0)
        x_min, y_min, x_max, y_max = map(int, box['coordinate'])
        
        # Lấy màu theo label, mặc định xám nếu label lạ
        color = LABEL_COLORS.get(label, (128, 128, 128))
        
        # Vẽ bounding box
        cv2.rectangle(img_viz, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Tạo text label với score
        label_text = f"{label} ({score:.2f})" if score else label
        
        # Vẽ nền cho text để dễ đọc
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_viz, (x_min, y_min - text_h - baseline - 4), (x_min + text_w, y_min), color, -1)
        cv2.putText(img_viz, label_text, (x_min, y_min - baseline - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    output_path = os.path.join(save_dir, f"layout_page_{page_idx + 1}.png")
    cv2.imwrite(output_path, img_viz)
    print(f"  [VIZ] Đã lưu visualization: {output_path}")
    return output_path

# ==========================================
# 3. HÀM GOM NHÓM LOGIC (Chỉ xử lý text, bỏ paragraph_title)
# ==========================================
def group_layout_boxes(boxes, center_tolerance=50):
    """
    Groups only 'text' boxes into column-clusters.
    Within each group, boxes are sorted top-to-bottom by y_min.
    Groups are sorted left-to-right by x_min.
    """
    text_boxes = [b for b in boxes if b['label'] == 'text']
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

    # BƯỚC 2: SẮP XẾP BOXES TRONG MỖI GROUP THEO Y (trên xuống dưới)
    for group in groups:
        group['boxes'] = sorted(group['boxes'], key=lambda b: b['coordinate'][1])

    # Sắp xếp các cột từ Trái sang Phải
    return sorted(groups, key=lambda g: g['x_min'])

# ==========================================
# 4. HÀM CẮT VÀ OCR MỘT BOX ĐƠN LẺ
# ==========================================
def crop_and_ocr_box(box, page_idx, box_idx, is_pdf, doc=None, page=None, page_rect=None, img_cv2=None, pad=5):
    """
    Crop a single layout box and send to Mistral OCR.
    Returns the OCR text result.
    """
    coord = box['coordinate']
    
    if is_pdf:
        x1, y1 = coord[0] - pad, coord[1] - pad
        x2, y2 = coord[2] + pad, coord[3] + pad
        crop_rect = fitz.Rect(x1, y1, x2, y2) & page_rect
        
        if crop_rect.is_empty or not crop_rect.is_valid:
            return ""

        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
        new_page = new_doc[0]
        new_page.set_cropbox(crop_rect)
        
        file_bytes = new_doc.tobytes()
        new_doc.close()
        filename_gia = f"page_{page_idx+1}_box_{box_idx}.pdf"
    else:
        x1 = max(0, int(coord[0]) - pad)
        y1 = max(0, int(coord[1]) - pad)
        x2 = min(img_cv2.shape[1], int(coord[2]) + pad)
        y2 = min(img_cv2.shape[0], int(coord[3]) + pad)
        
        cropped_img = img_cv2[y1:y2, x1:x2]
        _, buffer = cv2.imencode('.png', cropped_img)
        file_bytes = buffer.tobytes()
        filename_gia = f"page_{page_idx+1}_box_{box_idx}.png"

    print(f"    -> Đang OCR {filename_gia} ({box['label']})...")
    text_result = call_mistral_ocr(file_bytes, filename_gia)
    return text_result

# ==========================================
# 5. HÀM XỬ LÝ CHÍNH (Tự động nhận diện PDF / Ảnh)
# ==========================================
def process_and_ocr_document(file_path, center_tolerance=50, visualize=True):
    """
    Process a document (PDF or image) through layout detection and individual box OCR.
    Only processes 'text' boxes. paragraph_title and other types are ignored.
    
    Returns:
        list of dicts, each containing:
        {
            "page": int,
            "group_idx": int,       # column index (left-to-right)
            "box_idx": int,         # box index within the column (top-to-bottom)
            "label": "text",
            "coordinate": [x_min, y_min, x_max, y_max],
            "ocr_text": "..."
        }
    """
    print(f"Đang phân tích layout cho: {file_path}")
    model = LayoutDetection(model_name="PP-DocLayout-L")
    
    is_pdf = file_path.lower().endswith('.pdf')
    all_results = []

    # Chuẩn bị dữ liệu lặp
    if is_pdf:
        doc = fitz.open(file_path)
        num_pages = len(doc)
    else:
        num_pages = 1
        doc = None
        img_orig = cv2.imread(file_path)
        if img_orig is None:
            print(f"Lỗi: Không thể đọc ảnh {file_path}")
            return []

    box_counter = 0  # Global counter for unique box filenames

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
            page = None
            page_rect = None
            img_cv2 = img_orig

        # --- B. CHẠY PADDLE ---
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

        # --- B2. VISUALIZE LAYOUT (tất cả boxes, trước khi lọc) ---
        if visualize:
            visualize_layout(img_cv2, boxes, page_idx)

        # --- C. GOM NHÓM (chỉ text) VÀ OCR TỪNG BOX ---
        groups = group_layout_boxes(boxes, center_tolerance)

        for col_idx, group in enumerate(groups):
            for box_in_group_idx, box in enumerate(group['boxes']):
                box_counter += 1
                ocr_text = crop_and_ocr_box(
                    box, page_idx, box_counter,
                    is_pdf, doc=doc, page=page, page_rect=page_rect, img_cv2=img_cv2
                )
                
                all_results.append({
                    "page": page_idx + 1,
                    "group_idx": col_idx,
                    "box_idx": box_in_group_idx,
                    "label": box['label'],
                    "coordinate": [float(x) for x in box['coordinate']],
                    "ocr_text": ocr_text.strip()
                })

    if is_pdf:
        doc.close()
    
    # --- D. LƯU KẾT QUẢ (JSON có cấu trúc) ---
    output_json_path = "ket_qua_ocr_structured.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n========= HOÀN THÀNH =========")
    print(f"Đã lưu kết quả có cấu trúc vào: {output_json_path}")
    print(f"Tổng số text boxes đã OCR: {len(all_results)}")
    
    return all_results

# --- CHẠY THỬ ---
if __name__ == "__main__":
    # Đệ có thể truyền vào file .pdf hoặc .png/.jpg đều chạy mượt!
    # file_path = "image.png" 
    file_path = "/AIClub_NAS/WorkingSpace/Personal/chinhnm/HLLXRD/se_x_hllxrd/image.png" 
    results = process_and_ocr_document(file_path)
    
    # In kết quả tóm tắt
    for r in results:
        text_preview = r['ocr_text'][:80] + "..." if len(r['ocr_text']) > 80 else r['ocr_text']
        print(f"  Trang {r['page']} | Cột {r['group_idx']} | Box {r['box_idx']}: "
              f"\"{text_preview}\"")
