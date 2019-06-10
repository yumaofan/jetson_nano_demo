# jetson_nano_demo
Some great implement of deep learning algorithm in Nvidia jetson nano platform.
## Tasks
- [X] Real-time face detection
- [X] Real-time face recognition
- [ ] Real-time object detection
- [ ] Real-time object tracking
- [ ] Robot basic operation

## Dependencies
- Tensorflow 1.13 [[install]](https://devtalk.nvidia.com/default/topic/1048776/official-tensorflow-for-jetson-nano-/)
- Python 3.x
- Numpy
- sklearn
- scipy

If you have some errors when install packages using pip3,
please instead of using "sudo apt-get install python3-package_name" [refer](https://devtalk.nvidia.com/default/topic/1050614/jetson-nano/cannot-import-scipy-on-jetson-nano/)

## face recognition
1. Copy yourself face images into face_db folder, insure only one person in one image
2. python3 face_recognition/canera_demo.py

## References
[MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow)  
[MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF)
