# gaze_detect
CSCE 643 - Fall 2021
Eye gaze detection using neural networks.
LeNet-based model used to train on the MPIIGAZE dataset.  Head pose-independent and dependent gaze estimation is implemented.

Trained model is then utilized for webcam gaze inference using a 6-point face model and Perspective-n-Point solver. 

Also see `project_report.pdf`.


## Environment
Operating system: Ubuntu 21.04
Programming Language: Python 3.9.7

## Setup
1. Clone repository
    `git clone` 
2. Install requirements
    `pip install -r requirements.txt`
	
## Run inference
1. Download Dlib trained face detection model from https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2 and place it in the `assets/` directory.
2. Switch to the `code` directory:
	`cd code/`
2. Calibrate camera matrix
    `python calibrate_cam.py`
3. Run webcam inference code
    `python webcam_gaze.py`
    - To use head pose dependent model, modifications in utils_webcam.py's estimate_gaze and webcam_gaze.py's model definitions needed.

## Train model
1. Download MPIIGAZE dataset from http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz and place the extracted directory `MPIIGAZE` along with its contents in  the`data/` directory.
2. Switch to the `code` directory:
	`cd code/`
3. Set training configuration as desired in `train.py`
4. Train the model:
	`python train.py`
	- The trained model will overwrite the existing pretrained model in `assets/models/` unless the path is changed.
	- To train other models, use the following:
		1. Head pose-dependent LeNet:
		`python train_other_LeNet.py`
		2. Head pose-dependent AlexNet:
		`python train_other_AlexNet.py`

## References:
- Normalization: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/
- Other implementation: https://github.com/hysts/pytorch_mpiigaze/
