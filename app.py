from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Charger le modèle YOLO pour la détection de poses
model = YOLO('yolov8s-pose.pt', task="pose")

# Initialiser la webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Traiter la frame avec YOLO
        results = model(frame, stream=True, imgsz=640)

        for result in results:
            img = result.orig_img
            boxes = result.boxes
            try:
                for box in boxes:
                    x, y, w, h = box.xywh[0]
                    kpts = result.keypoints

                    # Dessiner les keypoints
                    if kpts is not None:
                        for keypoint in kpts.xy[0]:
                            kp_x, kp_y = int(keypoint[0].item()), int(keypoint[1].item())
                            cv2.circle(img, (kp_x, kp_y), 5, (0, 255, 0), -1)

                    # Annoter la chute ou la stabilité
                    if w / h > 1.4:
                        cv2.putText(img, "Fallen", (int(x), int(y - h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(img, "Stable", (int(x), int(y - h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print(f"Erreur : {e}")

        # Encoder les frames en JPEG
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Page avec la vidéo

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)