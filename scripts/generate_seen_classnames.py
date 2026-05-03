import os
import argparse
from pathlib import Path

# Danh sách UNSEEN_CLASSES được copy từ src/dataset_retrieval.py
UNSEEN_CLASSES = [
    "bat", "cabin", "cow", "dolphin", "door", "giraffe", "helicopter",
    "mouse", "pear", "raccoon", "rhinoceros", "saw", "scissors",
    "seagull", "skyscraper", "songbird", "sword", "tree", "wheelchair",
    "windmill", "window"
]

def generate_seen_list(data_dir, outfile):
    """
    Tạo file classnames chứa các lớp 'seen' bằng cách
    lấy tất cả các lớp và loại trừ đi các lớp 'unseen'.
    """
    sketch_dir = Path(data_dir) / 'sketch'
    
    if not sketch_dir.is_dir():
        print(f"Lỗi: Không tìm thấy thư mục '{sketch_dir}'.")
        print("Vui lòng cung cấp đường dẫn đúng đến thư mục gốc của dataset Sketchy.")
        return

    # 1. Lấy tất cả các class
    all_categories = sorted([d.name for d in sketch_dir.iterdir() if d.is_dir()])
    
    # 2. Loại bỏ các unseen classes
    seen_classes = sorted(list(set(all_categories) - set(UNSEEN_CLASSES)))
    
    # 3. Ghi ra file
    with open(outfile, 'w') as f:
        for class_name in seen_classes:
            f.write(f"{class_name}\n")
            
    print(f"Đã tạo thành công file '{outfile}' với {len(seen_classes)} seen classes.")
    print(f"Các class bị loại bỏ (unseen): {len(UNSEEN_CLASSES)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a list of 'seen' classes for the Sketchy dataset.")
    parser.add_argument('--data-dir', required=True, help="Đường dẫn đến thư mục gốc của dataset Sketchy (ví dụ: 'data/Sketchy').")
    parser.add_argument('--outfile', default='data/seen_classnames.txt', help="Đường dẫn file output để lưu danh sách seen classes.")
    args = parser.parse_args()
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(Path(args.outfile).parent, exist_ok=True)
    
    generate_seen_list(args.data_dir, args.outfile)
