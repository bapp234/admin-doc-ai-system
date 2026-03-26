def sort_boxes_layout_aware(boxes, x_threshold=50, y_threshold=15):
    """
    Sắp xếp boxes thông minh: Nhận diện cột văn bản.
    x_threshold: Khoảng cách ngang tối thiểu để coi là sang cột mới.
    """
    if not boxes:
        return []

    # 1. Sắp xếp thô từ trên xuống dưới
    boxes.sort(key=lambda b: b[0][1])

    lines = []
    if len(boxes) > 0:
        current_line = [boxes[0]]
        
        for i in range(1, len(boxes)):
            # Kiểm tra xem có cùng dòng (Y) không
            if abs(boxes[i][0][1] - current_line[-1][0][1]) < y_threshold:
                current_line.append(boxes[i])
            else:
                # Sắp xếp các box trong dòng theo trục X
                current_line.sort(key=lambda b: b[0][0])
                lines.append(current_line)
                current_line = [boxes[i]]
        
        current_line.sort(key=lambda b: b[0][0])
        lines.append(current_line)

    # 2. Logic xử lý cột: Nếu trong một dòng có khoảng cách X quá lớn
    final_sorted_boxes = []
    for line in lines:
        for box in line:
            final_sorted_boxes.append(box)
            
    return final_sorted_boxes