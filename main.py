import cv2
import numpy as np

# Web Camera
cap = cv2.VideoCapture('video.mp4')

count_line_position = 550
min_width_rect, min_height_rect = 80, 80
detections = []
in_counter = 0
out_counter = 0

# Initialize Subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = h //2
    cx = x + x1
    cy = y + y1
    return cx, cy

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 5)
    # Applying on all the frames
    img_sub = algo.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5),)
    dilata = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilata = cv2.morphologyEx(dilata, cv2.MORPH_CLOSE, kernel)
    contourShape, _ = cv2.findContours(dilata, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (0,0,255), 3)
    
    new_detections = []
    
    for (i, channel) in enumerate(contourShape):
        (x, y, w, h) = cv2.boundingRect(channel)
        if w >= min_width_rect and h >= min_height_rect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = center_handle(x, y, w, h)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            new_detections.append(center)
    
    for (x, y) in new_detections:
        for (prev_x, prev_y) in detections:
            if abs(x - prev_x) < min_width_rect and abs(y - prev_y) < min_height_rect:
                if prev_y > count_line_position >= y:
                    in_counter += 1
                if prev_y < count_line_position <= y:
                    out_counter += 1
    
    detections = new_detections
    
    cv2.putText(frame, f'In: {in_counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Out: {out_counter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('Original Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
