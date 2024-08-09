Object counting and tracking is s computer algorithm where we try to detect and count the total number of objects in a given image or a video file accurately and automatically. With advancements in technology, there has been a rise in the number
of object counting algorithms and technique. With recent increase in interest in the field of Machine Learning and Artificial Intelligence, object counting and tracking research has also shown remarkable growth with countless authors publishing their works.
 
With the introduction of Segment Anything Model(SAM) by META last year, a new branch in object counting space has grown. In this task, we use this newly introduced Segment Anything Model and perform the counting task. Segment Anything Model by META is an image segmentation model that can perform high quality masks from any given set of prompts (inputs) using promptable segmentation.

These prompts can range from a simple point to boxes and can be used to mask specific objects in an image file. In this novel approach, we combine Segment Anything Model with a detection layer of YOLO(You Only Look Once) to achieve greater results with more accuracy and prediction. YOLO is a real time is a single stage objectdetection algorithm introduced in 2015 to perform object detection tasks with more accuracy, more precision and also be time efficient. The aim of this research is to find the least number of images required using YOLO trained model to get better results than previous works.
![SAM + YOLO Diagram](https://github.com/user-attachments/assets/567a6a47-0e88-4f68-886f-d0c104931a01)

Results 
![SAM + YOLO Diagram2](https://github.com/user-attachments/assets/65a587f4-5c1e-4d7e-9b28-11adb2a973f4)

![SAM+YOLO9](https://github.com/user-attachments/assets/2a7ad2c2-c8b3-4ac7-a288-a9829ddc9357)
