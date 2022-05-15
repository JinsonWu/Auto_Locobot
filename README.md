# Auto-Driving_Implementation_on_LOCOBOT
Autonomous driving system is what people have dedicated to in past ten years. Nvidia and Tesla attempt to devise a absolutely non-manual driving experience on the road. Embedded systems application with Machine Learning algorithms are the mainstream in this domain. This project I utilized LOCOBOT with ROS system to implement auto-driving with the help of ResNet34. It in return achieved 85% of accuracy in 0.5sec response time. 

## Model Training
This model could be defined as multi-class image classification. Turning movement between a specific degree range was assigned an unique class, e.g. 0.1 rad turning was class L2 (left-second). Training was completed in personal computer, then output the checkpoints to LOCOBOT to do auto-driving implementation.  

### Dataset
Training and testing dataset are composed of 20 directions that 10 directions either left or right. There were about 50 images per class for training and 20 per class for testing.

### Training Procedure
Epoch is set to be 100. The optimal checkpoint would be automatically stored based on the loss function (entropy loss in this case). In this project, I used epoch_74 as the final checkpoint. 
  
*path files: https://drive.google.com/drive/folders/1BhuKuXj7PBKo2sawmxqQX_GNNvE87sAT?usp=sharing*
  
### Testing Result
Reached 85% of accuracy and responded in milliseconds.

## Contact Info
Author: Chun-Sheng Wu, MS student in Computer Engineering @ Texas A&M University  
Email: jinsonwu@tamu.edu  
LinkedIn: https://www.linkedin.com/in/chunshengwu/


