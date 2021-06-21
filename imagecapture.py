import cv2
import os

def start_capture():
    cwd = os.getcwd()
    
    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("Calibration Window")
    
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        cv2.imshow("Calibration Window", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:
            # ESC pressed
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "calib_{}.png".format(img_counter)
            data_path = cwd + "/data"
            fullpath = os.path.join(data_path, img_name)
            cv2.imwrite(fullpath, frame)
            
            img_counter += 1
    
    cam.release()
    
    cv2.destroyAllWindows()