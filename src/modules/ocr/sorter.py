def sort_boxes_layout_aware(boxes, x_threshold=50, y_threshold=15):
    if not boxes: return []
    boxes.sort(key=lambda b: b[0][1])
    
    lines, current_line = [], [boxes[0]]
    for i in range(1, len(boxes)):
        if abs(boxes[i][0][1] - current_line[-1][0][1]) < y_threshold:
            current_line.append(boxes[i])
        else:
            current_line.sort(key=lambda b: b[0][0])
            lines.append(current_line)
            current_line = [boxes[i]]
            
    current_line.sort(key=lambda b: b[0][0])
    lines.append(current_line)
    
    return [box for line in lines for box in line]