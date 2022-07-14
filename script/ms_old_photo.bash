#!/bin/bash
pwd
# pull the main repo
git clone https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life.git photo_restoration
# pull the syncBN repo
cd photo_restoration/Face_Enhancement/models/networks 
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch 
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm . 
cd ../../../
# pull the syncBN repo 2
cd Global/detection_models 
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch 
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm . 
cd ../../
# download the landmark detection model
cd Face_Detection/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
cd ../
# download the pretrained model
cd Face_Enhancement/
wget https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Face_Enhancement/checkpoints.zip
unzip checkpoints.zip
cd ../
# download the pretrained model
cd Global/
wget https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Global/checkpoints.zip
unzip checkpoints.zip
cd ../
pwd