# Soccer Players Tracking
![](https://datasets.simula.no/downloads/alfheim/2013-11-28/thumb/2013-11-28_thumb.png)

In this project we aim to reconstruct an soccer game's details from the position of the players and referee to the their movements using three recorded videos with different field coverage. Subsequently, movement and position of the individuals are displayed in a top-view demonstration of a 2D socccer pitch. For this project, we used "Soccer video and player position dataset" from [this website](https://datasets.simula.no/alfheim/).

### Input data
"Soccer video and player position dataset" provides three videos, each displaying a constant soccer match synchronously from a different prespective.

| Left | Center | Right |
| :---: | :---: | :---: |
| <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/assets/inputs/0.gif?raw=true" width="300px"> | <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/assets/inputs/1.gif?raw=true" width="300px"> | <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/assets/inputs/2.gif?raw=true" width="300px"> |


### Extracting background of the inputs
We extract the background of the videos by calculating the mean of all frames in the videos for each pixel.

| Left | Center | Right |
| :---: | :---: | :---: |
| <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/bg_0.png?raw=true" width="300px"> | <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/bg_1.png?raw=true" width="300px"> | <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/bg_2.png?raw=true" width="300px"> |

### Substracting the background
In this section, the obtained backgounds are used in a KNN Background Subtraction algorithm to detect moving individuals.

<img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/assets/demo/bg-sub.gif?raw=true" width="800px">

### Objects to Patches
The detected objects are converted to patches of variable size and saved in a folder named "img". Moreover, with the help of `pigeon.anotate` a correponding label is assigned to all of the patches.

| Patches |
| :---: |
| <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_0-60-6.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_0-60-7.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_0-60-8.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_0-140-42.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_0-60-10.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_0-80-14.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_0-80-16.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_0-80-13.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_0-80-17.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_0-100-23.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_0-80-19.jpg?raw=true" height="50px">  |


### Classifying Individuals
After extracting patches and labalizing them, we use 2 convolution layers following a flatten layer and 2 fully-connected layers to classify the patches into 3 classes.

| Blue Team | White Team | Referees |
| :---: | :---: | :---: |
| <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_0-100-23.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_1-1920-1399.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_1-1600-1199.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_1-1860-1349.jpg?raw=true" height="50px"> | <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_1-820-605.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_1-1600-1200.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_1-1580-1192.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_1-1560-1178.jpg?raw=true" height="50px"> | <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_1-1460-1117.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_1-1720-1254.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_1-1380-1052.jpg?raw=true" height="50px"> <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/img/0_1-2000-1470.jpg?raw=true" height="50px"> |

| Layer (type) | Output Shape |  
| --- | --- |
| conv2d (Conv2D) | (None, 26, 8, 64) |    
| max_pooling2d (MaxPooling2D) | (None, 13, 4, 64) |       
| conv2d_1 (Conv2D) | (None, 11, 2, 128) |  
| max_pooling2d_1 (MaxPooling2) | (None, 5, 1, 128) |
| flatten (Flatten) | (None, 640) |
| dense (Dense) | (None, 128) |
| dense_1 (Dense) | (None, 3) |

### Applying Masks
We apply two types of masks for different purposes on the input images.

#### Region of Interest (ROI)
This mask is used to define the region of interest in the image. Using this mask, we are able to omit the improper regions, like the big monitor in the left video, pitch-side hoardings, and audience. 
| Left | Center | Right |
| :---: | :---: | :---: |
| <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/assets/masks/mask0.png?raw=true" width="300px"> | <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/assets/masks/mask1.png?raw=true" width="300px"> | <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/assets/masks/mask2.png?raw=true" width="300px"> |

#### Top-view Coverage Area
We used this mask to define a unique top-view area for each input video. By applying this mask, all of the common areas between input videos will be eliminated.
| Left | Center | Right |
| :---: | :---: | :---: |
| <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/assets/masks/left.png?raw=true" width="300px"> | <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/assets/masks/center.png?raw=true" width="300px"> | <img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/assets/masks/right.png?raw=true" width="300px"> |

### Transformation
We transform the masked input videos and their objects to get three complementary top-view presentations.

<img src="https://github.com/TroddenSpade/Soccer-Players-Tracking/blob/main/assets/demo/transform.gif?raw=true" width="800px">

### Final results

## References
1. S. A. Pettersen, D. Johansen, H. Johansen, V. Berg-Johansen, V. R. Gaddam, A. Mortensen, R. Langseth, C. Griwodz H. K. Stensland, and P. Halvorsen, Soccer video and player position dataset, Proceedings of ACM MMSys 2014, March 19.