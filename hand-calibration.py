import cv2

def draw_rect(frame):
    rows, cols, _ = frame.shape
    
    cv2.rectangle(frame, (int(rows/2 - 5), int(cols/2 - 5)), 
                  (int(rows/2 + 5), int(cols/2 + 5)),
                  (0, 0, 255), 1)
    return frame

def hand_historgram(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros((10, 10, 3), dtype=hsv_frame.dtype)
    
    roi[:] = hsv_frame[int(rows/2 - 5):int(rows/2 + 5),
                       int(cols/2 - 5):int(cols/2 + 5)]
    
    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProjection([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)
    
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    
    thresh = cv2.merge((thresh, thresh, thresh))
    
    return cv2.bitwise_and(frame, thresh)