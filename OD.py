import cv2
from numba import jit, cuda
weightsPath = "frozen_inference_graph.pb"
configPath ="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt" 
model = cv2.dnn_DetectionModel(weightsPath,configPath)
classLabels = []
file_name = "labels.txt"
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320,320)
# model.setInputSize(50,50)
# model.setInputSize(720,720)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open the webcam")
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
while True:
    ret,frame = cap.read()
    # frame = cv2.resize(frame,( 360,240))
    frame = cv2.resize(frame,( 800,600))
    ClassIndex, confidece, bbox = model.detect(frame, confThreshold = 0.45)
    print(ClassIndex)
    if (len(ClassIndex)!= len(classLabels)):
        if len(ClassIndex) > 0:
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(),bbox):
                if(ClassInd<= 80):
                    cv2.rectangle(frame,boxes,(255,0,0),2)
                    cv2.putText(frame,classLabels[ClassInd -1],(boxes[0]+10, boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness =3)
                    # cv2.putText(frame,classLabels[1],(boxes[0]+10, boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness =3)
    cv2.imshow("Object Detection Tutorial",frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllwindows()
