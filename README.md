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

[image8]: ./pics/sliding_windows.PNG
[image9]: ./pics/hot_windows.PNG
[image10]: ./pics/heat_view.PNG
[image11]: ./pics/big_bbox.PNG



---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The idea here is as follows:

1. Get an image from the set of training images
2. Perform some data augmentation on the training set - this step is optional as it seems like many people have been able to get this project working without doing this but I found it helpful in reducing false detections
3. Convert image colorspace to one that you feel will yield a lot of features
4. Extract spatial features - resize each color channel of the input image to a predetermined size (this size is one of our tunable parameters) - append this feature to our main list of features 
5. Extract histogram features - take a histogram of each color channel of the input image with a predetermined number of histogram bins (this is one of our tunable parameters) - append this feature to our main list of features
6. Extarct HOG features - run the HOG algorithm on each color channel of the input image with predetermined parameters (`orient`, `pix_per_cell`, `cell_per_block` - these are tunable parameters) - append the features to our main list of features
7. Return the final feature list

The code for this step is starting at line 52 (line number subject to change) of the file called `lesson_functions.py`.

```python
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, notcar=False):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    np_images = []
    for img in imgs:
        # Read in each one by one
        #image = mpimg.imread(file)
        temp = cv2.imread(img)
        np_images.append(temp)
        if notcar:
            np_images.append(cv2.flip(temp,1))
            blur_shape = random.randint(1,15)
            if blur_shape %2 == 0: blur_shape -= 1
            blurred = cv2.GaussianBlur(temp, (blur_shape, blur_shape), 0)
            np_images.append(blurred)
            # Add noisy image
            #for j in range(random.randint(1,10)):
            noise = np.zeros_like(temp)
            cv2.randn(noise,(random.randint(0,10)),(random.randint(15,25)))
            noisy_img = temp+noise
            np_images.append(noisy_img)
    
    for image in np_images:
        # apply color conversion if other than 'RGB'
        file_features = []
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
 ```

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters to get the right ones. This was a very difficult and time consuming step. There were many different parameters to tune to be random about it. Getting these parameters right is critical in avoiding false positives and false negatives.

##### Colorspace

One of the most impactful changes was to use the `YUV` color space. Before making this change, I found it very hard to detect the white car. I noticed that there were some hot windows even using the default color space but they was not enough heat around the white car to be able to detect it while simultaneously rejecting false positives. The white car is that it is very bright and stands out in the Y channel.

YUV is a luma-chroma encoding scheme, meaning that color is deifined via one luminance value and two chrominance values.The Y component determines the brightness of the color (the luma), while the U and V components determine the color itself (the chroma). YUV is also in a way similar to human vision – the "black and white" information has more impact on the image for human eye than the color information. It's possible that some of the other luma-chroma schemes could have worked well. 

##### Other Parameters

My guiding philosophy for tuning the rest of the features was to try to keep them as small as possible while still maintaining good results. These parameters have a direct impact on training and inference speed and I didn't want to spend more time on these steps than necessary to produce a good result.

#### Final Parameters

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

I created a `train` method inside my class for the video processor which is called by the `__init__` method only if a pickled SVM was not already found. This currently starts at line 69 of `project.py`. 

The steps are:

1. Check if a trained SVM already exists - if yes, no need to retrain
2. If training is needed, extract features from car and non-car images
3. Create a vector of features
4. Create a vector of labels (1 if car, 0 if not car)
5. Split these vectors into randomized training and test data sets using `sklearn`
6. Ensure that all the features are normalized before training and we can use the `StandardScaler` in `sklearn` to do this.
7. Fit a linear SVC on the trainin

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window size was also a very difficult parameter to tune. One of the issues here was that choosing too many, too small, or too overlapping windows could DRASTICALLY slow down the image processing time. This really hampered my ability to do rapid iteration and experimentation. Using the wrong set of parameters could result in 0.5 FPS or even less. This means running through the entire video clip could take an hour...the HOG algorithm is aptly named for it is truly a hog!

At first, I tried the multiscale window approach but I became thoroughly frustrated after I realized that the processing time using this method was much too slow for me to be able to experiment with all of the various parameters. It probably would have been better for me to try the single scale approach _before_ moving on to more complex techniques but I tend to get excited about things and jump the gun at times.

Another issue that caused me much frustration was trying to detect vehciles on the opposite side of the highway. I had way too many false positives and false negatives that I could not filter out. Actually, I bet this is where I spent the **majority** of time in this project (don't ask how long :) )! The problem is that detecting cars on the southbound side of I-280 requires a larger ROI _and_ the use of multiscale windows which decreases FPS and thus the ability to experiment with parametrers _even further_. The cars on that side of the road are too small and move too fast to be able to use the same techniques. Again, it would have been much better to have started small before biting off too much.

Ultimatley, I settled on larger single scale 128x128 windows with 90% overlap in both the X and Y planes. I found this windows configuration to give me a good balance between FPS and ability to produce enough heat on true positives while negating false positives.

![alt text][image8]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on a single scale using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

To improve the performnce of the classfier, I augmented the dataset with some transform versions of the training data.

Here are some example images:

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

I am pretty satisfied with my result overall except for two things:

1. One false negative - the bounding box around the white car disappears for just a couple seconds before the black car pulls up behind it. 
2. One false positive - as the black car begins to pull ahead of the white car, my code encapsulates them in one big bounding box as if they were one vehicle rather than two distanct vehicles. This means that there is some false positive heat in the road space between the two cars. The box eventually splits in two once the black car pulls far ahead enough of the white car.

Here's a picture of the large bounding box.

![alt text][image11]

I tried to tune these issues like crazy but couldn't fix it. I am planning on going back and trying it with a different luma-chroma color system to see if this makes any difference. 

Another possible method to fix the large bounding box might be to specify a maximum pixel width for heatmaps/boxes. For any heatmap that exceeds this maximum, split it in two and zero out the middle pixels. This way, it would get picked up as two seperate labels and thus get two seperate bounding boxes.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The overall flow for detecting vehicles goes a little something like this:

1. Run the HOG on all windows
2. If a window is hot (meaning the SVM thinks it saw a vehicle in that window), save it to a list or queue
3. Take the sum of all the heatmaps in the list/queue. If enough overlapping windows are hot, the overlapping region gets even hotter - we ideally want it to be engulfed in flames!
4. Apply a heat threshold to filter out the less hot regions so we're only left with the scorching hot ones
5. Now all the contiguously hot region "blobs" can be considered to be a vehicle - we call these blobs "labels"
6. Take the pixels positions of a box surrounding the hot regions and use them to draw a bounding box in the original image

In a perfect world, this would be enough...but then again in a perfect wrold, cars would already be driving themselves too. The issue is that false positives still pop up. A false positive is a box around anything that is _not_ a vehicle.

Lucky for me, by the time I had got to this point, I had tuned my HOG parameters and window sizes such that I had minimal false positives. The ones that remained had relatively **small** bounding boxes in comparison to the bounding boxes around the true positives. All I had to do to get rid of these small false positives was to calculate the area (width x height) of each bounding box before drawing it and only keep the boxes that were of a minimum size. I found 2000 pixels squared to be a good threshold. After doing this, no more false positives in my output! 

```python
# Draw the box on the image
area = (bbox[1][0] - bbox[0][0])*(bbox[1][1] - bbox[0][1])
if area > 2000:
    cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    bboxes[car_number]=bbox
```

Also, I was able to make my bounding boxes smooth and not jittery by averaging the vertex positions of the last frame's bounding box for each vehicle with the current frame's bounding box vertices.

```python
if car_number in old_boxes:
    bbox = (
            ((np.min(nonzerox)+old_boxes[car_number][0][0])//2, (np.min(nonzeroy)+old_boxes[car_number][0][1])//2),
            ((np.max(nonzerox+old_boxes[car_number][1][0])//2), (np.max(nonzeroy)+old_boxes[car_number][1][1])//2)
    )
else:
    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
```

### Here are is an example of the heatmaps queue summed up over 10 frames:

![alt text][image10]

You can see that the pixels are quite hot. This is due to the sliding windows overlap that I chose (90% in X and Y directions) and also the fact that I chose to add and extra heat value to pixels that already have previous heat in them so as to make the hot regions stand out better. This is what I am talking about. Notice the extra lines to make alread hot pixel even hotter.

```python
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        if np.max(heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]]) > 1:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
```

---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

##### Pain Points

Like the projects before this one, this pipeline is hardly robust. All the same reasons that I have already yammered on about (lighting conditions, road conditions, curvature, etc) in previous write ups also apply here. I guess the unique thing in this project is the issue of defining a good region of interest. For example, what if the car is driving up or down a steep hill in a place like San Francisco? Then the ROI would need to be extended upwards into what normally would be considered the sky. What if the car is standing stationary at the peak of a hill looking downwards? Then the ROI would need to be cut down drastically or else we risk getting false positives from clouds that look like cars and driving off a cliff! Maybe that's a bit dramatic but you get the idea. Part of the answer to this problem is that perception is not the entirety of autonomous driving. There is HD mapping, localization, path planning, and sensor diversity and fusion which also play a part in averting disaster. 

##### Processing Speed

I've said it before and I'll say it again - the biggest problem in this lab exercise was the slow processing time! It all seems so easy in retrospect but when you're starting off, you have no idea what you're doing and just want to go through the process of trial and error (and least I did). The SVM/HOG is not a technique which lends itself easily to trial and error addicts like me. One technique I used which did help speed things up a little bit was to cache my SVM and hot windows in a pickle file. This was useful once I got to the point where I was fine tuning the process of filtering my heat maps. However it was not useful for tuning HOG parameters or sliding windows sizes since altering those requires regenrating the SVM and hot windows.

##### Debug Visualizations

The thing that helped me understand the theory of what was going on under the hood and helped me fine tune my piepline the most was the alternative debug visualizations which I created. I made a bunch of hotkeys that would dynamically toggle the sliding windows, hot windows, and heatmaps on the live streaming video. These were a life saver for me.

##### Shoulda', Coulda', Woulda' Used a CNN

To speed things up, I would go back an implement this lab with a CNN instead of using an SVM to extract HOG features. The high level technique would have been the same which is to run the network over sliding windows and to aggregate the output. However, inference time would have been much less, training less complicated, and accuracy much higher. There are some networks out there like YOLO (You Only Look Once) which do simultaneous detection and localization but I don't know much about that technique...yet.

I honestly kind of wish I just used a CNN in the first place because it is more practical anyways. I was reading somewhere that Dr. Andrej Karpathy (the Yoda of CNNs) himself even says that using HOG for this purpose is so outdated that it is only studied to gain historical context (or however he said it). That means that although this lesson was tremendously enriching, it was still just a _bit_ irrelevant to have to learn so much about the HOG. But I digress....



