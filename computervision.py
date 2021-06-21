import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imagecapture
import cameracalibration
from collections import deque
import re

# for smoothing values
rowq = deque(maxlen=10)
pitchq = deque(maxlen=10)
yawq = deque(maxlen=10)


#convert rotation vector to roll pitch yaw
def rot_params_rv(rvecs):
    from math import pi,atan2,asin
    R = cv.Rodrigues(rvecs)[0]
    
    #do smoothing
    rowq.append(180*atan2(-R[1][0], R[0][0])/pi)
    pitchq.append(180*atan2(-R[2][1], R[2][2])/pi)
    yawq.append(180*asin(R[2][0])/pi)

    #rowl = list(rowq)
    roll = sum(rowq) / len(rowq)
    pitch = sum(pitchq) /  len(pitchq)
    yaw = sum(yawq) /  len(yawq)


    rot_params= [roll,pitch,yaw]
    return rot_params

def draw_info(x, y, roll, pitch, yaw, dimg):
    font                   = cv.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.8
    fontColor              = (255,255,255)
    lineType               = 2
    
    height, width, ch = dimg.shape
    mh = int(height/2)
    maxsep = int(mh/5)
    
    #ensure that the drawn lines are not outside
    sep = 50
    if sep > maxsep:
        sep = maxsep
    
    xt = 'X: %.2f' % x
    cv.putText(dimg,xt,  (10,mh),font, fontScale,fontColor,lineType)
    
    yt = 'Y: %.2f' % y
    cv.putText(dimg, yt, (10,mh + sep), font, fontScale,fontColor, lineType)

    rollt = 'Roll: %.2f' % roll
    cv.putText(dimg, rollt, (10,mh + sep * 2),font, fontScale, fontColor,lineType)
    
    pitcht = 'Pitch: %.2f' % pitch
    cv.putText(dimg,pitcht,  (10,mh + sep * 3), font, fontScale,fontColor,lineType)
    
    yawt = 'Yaw: %.2f' % yaw
    cv.putText(dimg,yawt,  (10,mh + sep * 4), font, fontScale, fontColor,lineType)

    return dimg


def find_matches_in_image(marker, img, kp1, des1, threshold):

    # Initiate ORB detector
    orb = cv.ORB_create()

    #simple img processing by converting to bw
    imgbw = cv.cvtColor(img, 0)
    kp2, des2 = orb.detectAndCompute(imgbw,None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    #should do a check for upper limit of matches

    return kp2, des2, matches[:threshold]


def detect_marker_in_2d_image(marker, img, kp1, des1, mtx, dist):  
    kp2, des2, matches = find_matches_in_image(marker, img, kp1, des1, 10)

    #create vector for only top 10 matches
    mkp = []
    ikp = []
    
    for m in matches:
       #get object points
       mkp.append(kp1[m.queryIdx])
       #get img points
       ikp.append(kp2[m.trainIdx])

    #parse object points and image points
    objpt = cv.KeyPoint.convert(mkp)
    imgpt = cv.KeyPoint.convert(ikp)
    
    #reshape objpt from 2d to 3d (since we are using 2d picture)
    objvec = []
    for i in objpt:
        objvec.append([i[0], i[1], 0])
     
    zobj = np.asarray(objvec)
    zimg = np.asarray(imgpt)
    
    # Find the rotation and translation vectors.
    _, rvecs, tvecs, inliers = cv.solvePnPRansac(zobj, zimg, mtx, dist)
    
    #convert rotation vector to row pitch yaw
    rpy= rot_params_rv(rvecs)

    #draw pointers
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    
    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
 
    #calculate center of points to draw the indicator
    x = 0
    y= 0
    for p in zimg:
        x = x + p[0]
        y = y+ p[1]
    
    x = x /10
    y = y/ 10
    center = (int(x),int(y))
    
    #draw matches in the image
    img = cv.drawMatches(marker,kp1,img,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
      
    #draw the information
    img = draw_info(x, y, rpy[0], rpy[1], rpy[2], img)
    
    
    return img


def image_marker_detection(markerPath, imgPath):    
    with np.load('./data/calib.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    imgmarker = cv.imread(markerPath)
    imgCol =  cv.imread(imgPath) # trainImage

    #simple img processing by converting to bw
    markerbw = cv.cvtColor(imgmarker, 0)
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(markerbw,None)
    # compute the descriptors with ORB
    kp1, des1 = orb.compute(markerbw, kp)

    imgWithInfo = detect_marker_in_2d_image(imgmarker,imgCol,kp1, des1,  mtx, dist)

    #first write to file
    cv.imwrite("assignment_result.png", imgWithInfo)

    #show image
    cv.imshow("Computer Vision Demo", imgWithInfo)
    k = cv.waitKey(0)
    if k%256 == 27 or k%256 == 32:
        cv.destroyAllWindows()


def video_marker_detection(markerPath):
    with np.load('./data/calib.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    
    print(markerPath)
    print("creating marker points")
    imgmarker = cv.imread(markerPath)

    #simple img processing by converting to bw
    markerbw = cv.cvtColor(imgmarker, 0)
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(markerbw,None)
    # compute the descriptors with ORB
    kp1, des1 = orb.compute(markerbw, kp)

    cam = cv.VideoCapture(0)
    cv.namedWindow("Computer Vision Demo")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        k = cv.waitKey(1)
        if k%256 == 27:
            break
     
        imgWithInfo = detect_marker_in_2d_image(imgmarker, frame, kp1, des1, mtx, dist)

        scale_percent = 80 # percent of original size
        width = int(imgWithInfo.shape[1] * scale_percent / 100)
        height = int(imgWithInfo.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resizedImg = cv.resize(imgWithInfo, dim, interpolation = cv.INTER_AREA)
 
        cv.imshow("Computer Vision Demo", resizedImg)

    cam.release()
    cv.destroyAllWindows()


def main():
    import sys
    choice = re.sub(r'\W+', '', sys.argv[1])
    if ((len(sys.argv)==2) and (choice == 'capture')):
        print("Start capture")
        imagecapture.start_capture()
    elif ((len(sys.argv)==2) and (choice == "calibrate")):
        print("Calibrate Cam")
        cameracalibration.calibrate_camera()   
    elif ((len(sys.argv)==4) and (choice == "solveimg")):
        #sys.argv[2] = specify image marker file
        #sys.argv[3] = specify test image file
        print("Start Img Detect", sys.argv[3])
        image_marker_detection(sys.argv[2], sys.argv[3])    
    
    elif ((len(sys.argv)==3) and (choice == 'solvevid')):
        #sys.argv[2] = specify image marker file
        print("Start Vid Detect", sys.argv[2])
        video_marker_detection(sys.argv[2])
        
if __name__ == "__main__":
    main()
