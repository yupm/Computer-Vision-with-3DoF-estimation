# Computer-Vision-with-3DoF-estimation


## INSTALLATION
First set up a Python 3.6 + based virtual environment.
Next install the required dependencies from the requirements.txt file.

```
pip install -r requirements.txt
```

## 1.1 IMAGE CAPTURE FOR CAMERA CALIBRATION
Please ensure that the computer has a web camera before running this command.
```
python computervision.py -capture
```
This will launch a window showing the video feed. Please hold the printed chessboard.png n a hard, straight surface, and capture 10-16 images by pressing the [Space] key.

To exit from the window, please press the [Esc] key.

## 1.2 CAMERA CALIBRATION
Next run this command after you have taken 10-16 photos from the previous instance.
```
python computervision.py –calibrate
```
This will automatically carry out image calibration. Please wait while all images are processed before going to the next step in 1.3.

## MARKER IDENTIFICATION IN IMAGE
The -solveimg argument accepts 2 arguments after input. The first argument is the path of the marker. The second argument is the path of the test image. This will generate the result assignment_result.png and display the image as well. Please press [Esc] to exit.
```
python computervision.py –solveimg [marker.png] [testimg.png]
e.g python computervision.py –solveimg "nerv.png" "test_img.png"
```

## MARKER IDENTIFICATION IN VIDEO
This will generate a video display providing constant estimation of the marker. Please press [Esc] to exit.
```
python computervision.py –solvevid nerv.png
```
![Image](/images/example.gif)
