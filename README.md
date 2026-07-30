# productivity-checker

This project uses a model trained in ```jetson-inference``` to measure productivity levels when I'm working. Since I get distracted very easily when I should be doing work, I thought this project would be both a great learning experience and a way to solve a real world problem that I encounter on a daily basis.

## Overview 

This project uses a resnet-18 model which was trained on 3 classes, sleeping, looking at phone, and working. The dataset used a custom script located in ```script.py``` that takes multiple pictures on a camera feed, and saves them into a specified directory. The model is trained on this data and productivity levels are measured based on the confidence of the model's inference.

<img width="393" height="308" alt="Screenshot from 2026-07-29 16-42-38" src="https://github.com/user-attachments/assets/59401dc3-1bd0-44fa-89b8-9af5344e9777" />
<img width="393" height="308" alt="Screenshot from 2026-07-29 16-42-10" src="https://github.com/user-attachments/assets/31635a3b-f146-4ab3-b3ed-8e6dfa5e40e7" />
<img width="393" height="308" alt="Screenshot from 2026-07-29 16-42-26" src="https://github.com/user-attachments/assets/73f6e8cf-4fc1-4025-bbe9-2960a9c32253" />


## Classification Model 

### Model Summary

<img width="1185" height="776" alt="Screenshot 2026-07-29 164012" src="https://github.com/user-attachments/assets/e9dcbfe4-5e5d-4ab2-9306-2baea5e8e51f" />

### Model Metrics

<img width="983" height="1147" alt="Screenshot 2026-07-29 163154" src="https://github.com/user-attachments/assets/2b8b1c37-d0d5-46a0-a07c-63d82f3231c7" />

## Set up

1. SSH into orin  
2. Install necessary packages  
```sudo apt-get update```  
```sudo apt-get install -y git cmake build-essential libpython3-dev python3-pip python3-numpy```
3. Clone jetson-inference  
```git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference```
4. Update submodules  
```git submodule update --init --recursive```
5. Remove npymath because of a compatibility issue  
```sed -i 's/ npymath//g' python/bindings/CMakeLists.txt```  
```sed -i 's/ npymath//g' utils/python/bindings/CMakeLists.txt```
6. Install cmake  
```mkdir build```  
```cd build```  
```cmake ../```

Clone the repository:
```git clone git@github.com:kezhang29/productivity-checkers.git```

Run the project:
```python3 model.py```

## Video Demonstration

[https://drive.google.com/drive/folders/1uqQTJoC8pheT3QXdLASYytw_R9MTS6oS?usp=drive_link](https://drive.google.com/drive/folders/1uqQTJoC8pheT3QXdLASYytw_R9MTS6oS?usp=drive_link)

## Next Steps
- The validation set resembles the training set too much, causing the steep jump in validation accuracy shown in tensorboard. Retraining with a more diverse dataset would improve model results.
- Add more labels 



