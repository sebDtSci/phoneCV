# from ultralytics import YOLO

# # model = YOLO('yolov8n.pt',task='detect')
# model = YOLO("yolov8m-seg.pt")
# Threshold=0.3
# model("0", conf=Threshold, show=True, max_det=1)

######################################
from ultralytics import YOLO
import cv2

model = YOLO('yolov8s-pose.pt',task="pose") 

results = model("0", stream=True, save=True, imgsz=640)
frame=1
fall=0

for result in results:
    img = result.orig_img
    try:
        boxes = result.boxes  
        for box in boxes:
            x = boxes.xywh[0][0]
            y = boxes.xywh[0][1]
            w = boxes.xywh[0][2]
            h = boxes.xywh[0][3]
            kpts = result.keypoints
            nk = kpts.shape[1]
            for i in range(nk):
                keypoint = kpts.xy[0, i]
                x, y = int(keypoint[0].item()), int(keypoint[1].item())
                #Draw keypoints on img
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

            if w/h > 1.4:
                fall+=1
                print("Fall detected at {} frame".format(frame))
                
                #Print fall on top of persons head (Fucking important to see fall!!!!!)
                cv2.putText(img, "Fallen", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                
            else:
                cv2.putText(img, "Stable", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite("frames/frame_{:04d}.jpg".format(frame), img)  # Use zero-padding with 4 digits

        cv2.imshow('Image', img)  
        cv2.waitKey(1)
    except:
        pass
    cv2.imwrite("frames/frame_{:04d}.jpg".format(frame), img)  # Use zero-padding with 4 digits
    frame += 1

cv2.destroyAllWindows() 
print("Total fall detected: {}".format(fall))