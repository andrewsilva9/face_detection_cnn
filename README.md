# face_detection_cnn

There's a short post about this repo on my site, which you can find at: http://www.andrew-silva.com/2017/01/face-detection-and-pose-with-caffe-in.html

Basically, you will need Caffe, numpy, and OpenCV (2.4.x should work, but if it helps I'm on 2.4.13). Then you have to find areas where I have hard-coded paths to my own Caffe installation of my own project repo (just search for caffe_model_path and caffe_root and change the directories accordingly). Finally, running demo.py should work and give you some faces on the images I've included for testing purposes. Running webcam_runner.py should allow you to run the detector live on your own webcam.
