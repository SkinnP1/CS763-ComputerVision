import sys
import cv2

video = sys.argv[1]
capture = cv2.VideoCapture(video)
w,h = 175,75

while(True):
    # Capture frame-by-frame
    ret, frame = capture.read()
    if frame is None :
        break
    height , width , channels = frame.shape
    x = width - w 
    y = 0
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), -1)
    # Add text
    cv2.putText(frame, "Piyush Onkar", (x + int(w/10),y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.rectangle(gray, (x, y), (x + w, y + h), (255,255,255), -1)
    # Add text
    cv2.putText(gray, "Piyush Onkar", (x + int(w/10),y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.imshow('Color', frame)
    cv2.imshow("Grayscale",gray)
    cv2.moveWindow('Color', 0, 0)
    cv2.moveWindow('Grayscale', 700, 0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()