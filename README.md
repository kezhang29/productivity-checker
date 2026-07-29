# classroom-focus

This project measures student engagement in classrooms in real-time. It uses a jetson orin and a webcam and features live student detection and engagement labels. 

<img width="912" height="696" alt="Screenshot 2026-07-28 113333" src="https://github.com/user-attachments/assets/cfd1923e-796e-4bb2-bb3a-dc6fcba8102d" />

## Overview 

Uses ssd mobilenet model to detect students and feed into a trained resnet18 model which labels the students. The resnet model performs a categorical classification task where it decides whether the student is working (engaged), or sleeping (not engaged) or looking at their phone (not engaged). The final productivity score is calculated by the percentage of total students that are engaged.


## Classification Model Metrics

<img width="350" height="350" alt="Screenshot 2026-07-28 105801" src="https://github.com/user-attachments/assets/f9edb7ed-9dc4-4b11-9a4d-a7374c00d4d1" />
<img width="350" height="350" alt="Screenshot 2026-07-28 105700" src="https://github.com/user-attachments/assets/b1774e49-56dc-4f32-9af7-c63fd5d518dc" />
<img width="350" height="350" alt="Screenshot 2026-07-28 105717" src="https://github.com/user-attachments/assets/72e4b641-5f01-4991-8666-27df9dbab9ca" />
<img width="350" height="350" alt="Screenshot 2026-07-28 105708" src="https://github.com/user-attachments/assets/96a6b68b-b9f1-4eaf-a574-ee62e6a80d09" />

## Set up

Train the model inside ```jetson-inference```.

Clone the repository:
```git clone git@github.com:kezhang29/classsroom-focus.git```

Run the project:
```python3 model.py```


## Future Improvements
- Support additional engagement classes
- Improve robustness under different classroom layouts
- Smooth  video frames
- Analytics dashboard with historical engagement statistics
- Support multiple cameras
