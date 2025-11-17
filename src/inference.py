import cv2
import numpy as np
from collections import Counter
from data import preprocess_image
import config


def predict_digit(model, img):
    img_flat = preprocess_image(img)
    return model.predict(img_flat)[0]


def draw_ui(frame, roi_resized, prediction, threshold, is_live):
    h, w = frame.shape[:2]
    display = np.zeros((h, w + 300, 3), dtype=np.uint8)
    display[:h, :w] = frame
    
    sx = w
    display[:, sx:] = (40, 40, 40)
    
    cv2.putText(display, "DIGIT RECOGNIZER", (sx + 15, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    
    status = "LIVE" if is_live else "STANDBY"
    color = (0, 255, 100) if is_live else (100, 100, 100)
    cv2.putText(display, status, (sx + 15, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    box_color = (0, 255, 100) if is_live and prediction else (100, 100, 100)
    cv2.rectangle(display, (sx + 10, 110), (w + 290, 200), box_color, 2)
    
    if is_live and prediction is not None:
        cv2.putText(display, "PREDICTION", (sx + 15, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, str(prediction), (sx + 90, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 100), 3)
    else:
        cv2.putText(display, "NO PREDICTION", (sx + 15, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    cv2.putText(display, f"Threshold: {threshold}", (sx + 15, 230),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
    
    if roi_resized is not None:
        roi_color = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)
        roi_scaled = cv2.resize(roi_color, (100, 100))
        display[260:360, sx+100:w+200] = roi_scaled
        cv2.putText(display, "ROI:", (sx + 15, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    
    controls = ["CONTROLS:", "i = Toggle", "q = Quit", "Slider = Threshold"]
    for i, text in enumerate(controls):
        cv2.putText(display, text, (sx + 15, 380 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return display




class WebcamRecognizer:
    def __init__(self, model, threshold_init=config.THRESHOLD_INIT):
        self.model = model
        self.threshold = threshold_init
        self.is_live = False
        self.prediction = None
        self.pred_history = []
        self.smooth_window = config.PREDICTION_SMOOTHING_WINDOW
        self.conf_threshold = config.PREDICTION_CONFIDENCE_THRESHOLD

    def _smooth_prediction(self, pred):
        if pred is None:
            return None
        self.pred_history.append(pred)
        if len(self.pred_history) > self.smooth_window:
            self.pred_history.pop(0)
        if len(self.pred_history) < 3:
            return None
        counter = Counter(self.pred_history)
        common_pred, count = counter.most_common(1)[0]
        confidence = count / len(self.pred_history)
        return common_pred if confidence >= self.conf_threshold else None

    def _extract_roi(self, binary_img):
        h, w = binary_img.shape
        cx, cy = w // 2, h // 2
        radius = 80
        
        y1, y2 = max(0, cy - radius), min(h, cy + radius)
        x1, x2 = max(0, cx - radius), min(w, cx + radius)
        roi = binary_img[y1:y2, x1:x2]
        
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return None
        
        if config.USE_MORPHOLOGY:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)
            roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam!")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        cv2.namedWindow('Digit Recognizer')
        cv2.setMouseCallback('Digit Recognizer', lambda *a: self._toggle_live())
        cv2.createTrackbar('Threshold', 'Digit Recognizer', self.threshold, 255,
                          lambda x: setattr(self, 'threshold', x))
        
        print("\n" + "="*60)
        print("WEBCAM STARTED")
        print("="*60)
        print("Controls: i=toggle | q=quit | Adjust threshold slider")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('i'):
                    self._toggle_live()
                elif key == ord('q'):
                    print("✓ Quitting...")
                    break
                
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                radius = 80
                
                color = (0, 255, 100) if self.is_live else (100, 100, 100)
                cv2.rectangle(frame, (cx - radius, cy - radius), 
                            (cx + radius, cy + radius), color, 3 if self.is_live else 2)
                
                roi_resized = None
                if self.is_live:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
                    
                    roi_resized = self._extract_roi(binary)
                    if roi_resized is not None:
                        raw_pred = predict_digit(self.model, roi_resized)
                        self.prediction = self._smooth_prediction(raw_pred)
                
                display = draw_ui(frame, roi_resized, self.prediction, 
                                self.threshold, self.is_live)
                cv2.imshow('Digit Recognizer', display)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Webcam closed.")
    
    def _toggle_live(self):
        self.is_live = not self.is_live
        self.pred_history = []
        self.prediction = None
        print(f"✓ Inference {'ON' if self.is_live else 'OFF'}")
