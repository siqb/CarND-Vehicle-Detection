## Real-time Vehicle Detection and Tracking

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  




I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...
This was a very difficult step. There were many different parameters to tune to be random about it

One of the most impactful changes was to use the `YUV` color space. Before making this change, I found it very hard to detect the white car. I noticed that there were some hot windows even using the default color space but they was not enough heat around the white car to be able to detect it while simultaneously rejecting false positives.

This is the final set of HOG parameters which I used:

```python
self.color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
self.orient = 11  # HOG orientations
self.pix_per_cell = 16 # HOG pixels per cell
self.cell_per_block = 2 # HOG cells per block
self.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
self.spatial_size = (8, 8) # Spatial binning dimensions
self.hist_bins = 8    # Number of histogram bins
```


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...




### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window size was also a very difficult parameter to tune. One of the issues here was that choosing too many, too small, or too overlapping windows could DRASTICALLY slow down the image processing time. This really hampered my ability to do rapid iteration and experimentation. Using the wrong set of parameters could result in 0.5 FPS or even less. This means running through the entire video clip could take an hour...the HOG algorithm is aptly named for it is truly a hog!

At first, I tried the multiscale window approach but I became thoroughly frustrated after I realized that the processing time using this method was much too slow for me to be able to experiment with all of the various parameters. It probably would have been better for me to try the single scale approach _before_ moving on to more complex techniques but I tend to get excited about things and jump the gun at times.

Another issue that caused me much frustration was trying to detect vehciles on the opposite side of the highway. I had way too many false positives and false negatives that I could not filter out. Actually, I bet this is where I spent the **majority** of time in this project (don't ask how long :) )! The problem is that detecting cars on the southbound side of I-280 requires a larger ROI _and_ the use of multiscale windows which decreases FPS and thus the ability to experiment with parametrers _even further_. The cars on that side of the road are too small and move too fast to be able to use the same techniques. Again, it would have been much better to have started small before biting off too much.

Ultimatley, I settled on larger single scale 128x128 windows with 90% overlap in both the X and Y planes. I found this windows configuration to give me a good balance between FPS and ability to produce enough heat on true positives while negating false positives.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The overall flow for detecting vehicles goes a little something like this:

1. Run the HOG on all windows
2. If a window is hot (meaning the SVM thinks it saw a vehicle in that window), save it to a list
3. If enough overlapping windows are hot, the overlapping region gets even hotter - we ideally want it to be engulfed in flames!
4. Apply a heat threshold to filter out the less hot regions so we're only left with the scorching hot ones
5. Now all the contiguously hot region "blobs" can be considered to be a vehicle
6. Take the pixels positions of a box surrounding the hot regions and use them to draw a bounding box in the original image

In a perfect world, this would be enough...but then again in a perfect wrold, cars would already be driving themselves too. The issue is that false positives still pop up. A false positive is a box around anything that is _not_ a vehicle.

Lucky for me, by the time I had got to this point, I had tuned my HOG parameters and window sizes such that I had minimal false positives. The ones that remained had relatively **small** bounding boxes in comparison to the bounding boxes around the true positives. All I had to do to get rid of these small false positives was to calculate the area (width x height) of each bounding box before drawing it and only keep the boxes that were of a minimum size. I found 2000 pixels squared to be a good threshold. After doing this, no more false positives in my output! Here's the code - super simple:

```python
def draw_labeled_bboxes(img, labels, old_boxes):
  ....
  ....
  ....
        # Draw the box on the image
        area = (bbox[1][0] - bbox[0][0])*(bbox[1][1] - bbox[0][1])
        if area > 2000:
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
            bboxes[car_number]=bbox
    # Return the image
    return img, bboxes
```

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

##### Pain Points

Like the projects before this one, this pipeline is hardly robust. All the same reasons that I have already yammered on about (lighting conditions, road conditions, curvature, etc) in previous write ups also apply here. 

##### Processing Speed

I've said it before and I'll say it again - the biggest problem in this lab exercise was the slow processing time! It all seems so easy in retrospect but when you're starting off, you have no idea what you're doing and just want to go through the process of trial and error (and least I did). The SVM/HOG is not a technique which lends itself easily to trial and error addicts like me. One technique I used which did help speed things up a little bit was to cache my SVM and hot windows in a pickle file. This was useful once I got to the point where I was fine tuning the process of filtering my heat maps. However it was not useful for tuning HOG parameters or sliding windows sizes since altering those requires regenrating the SVM and hot windows.

##### Shoulda', Coulda', Woulda' Used a CNN

To speed things up, I would go back an implement this lab with a CNN instead of using an SVM to extract HOG features. The high level technique would have been the same which is to run the network over sliding windows and to aggregate the output. However, inference time would have been much less, training less complicated, and accuracy much higher. There are some networks out there like YOLO (You Only Look Once) which do simultaneous detection and localization but I don't know much about that technique...yet.

I honestly kind of wish I just used a CNN in the first place because it is more practical anyways. I was reading somewhere that Dr. Andrej Karpathy (the Yoda of CNNs) himself even says that using HOG for this purpose is so outdated that it is only studied to gain historical context (or however he said it). That means that although this lesson was tremendously enriching, it was still just a _bit_ irrelevant to have to learn so much about the HOG. But I digress....



