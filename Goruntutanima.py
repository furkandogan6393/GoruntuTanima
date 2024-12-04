import cv2
import numpy as np

# Model dosyaları
model_config = r"C:\Users\TozLu\Desktop\Yapay Zeka\yolov4.cfg"  # YOLO config dosyası
model_weights = r"C:\Users\TozLu\Desktop\Yapay Zeka\yolov4.weights"  # YOLO önceden eğitilmiş ağırlıklar
classes_file = r"C:\Users\TozLu\Desktop\Yapay Zeka\coco.names"  # COCO veri kümesi etiketleri


with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


net = cv2.dnn.readNet(model_weights, model_config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break


    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)


    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)


                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (191, 255, 0) 
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), font, 1, color, 2)


    cv2.imshow("Nesne Tespiti", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
