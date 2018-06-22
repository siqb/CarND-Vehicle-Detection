import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from heat import *
from hog_subsample import *
from scipy.ndimage.measurements import label
import pickle
import pygame

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True,
                    window_size=(64,64)):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], window_size)      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

class MyVideoProcessor(object):

    # constructor function
    def __init__(self):
        import collections
        #self.heat_len = 40 
        #self.heat_threshhold =245 
        self.heat_len = 60 
        self.heat_threshhold =245 
        self.heatmaps = collections.deque(maxlen=self.heat_len)
        self.bboxes_prev_frame = {}
        self.prev_frame_labels = None, 0
        ### TODO: Tweak these parameters and see how the results change.
        self.color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        #self.orient = 9  # HOG orientations
        self.orient = 11  # HOG orientations
        #self.pix_per_cell = 8 # HOG pixels per cell
        self.pix_per_cell = 16 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        #self.hog_channel = 2 # Can be 0, 1, 2, or "ALL"
        self.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        #self.spatial_size = (16, 16) # Spatial binning dimensions
        self.spatial_size = (8, 8) # Spatial binning dimensions
        #self.spatial_size = (32, 32) # Spatial binning dimensions
        #self.hist_bins = 16    # Number of histogram bins
        self.hist_bins = 8    # Number of histogram bins
        #self.hist_bins = 32    # Number of histogram bins
        self.spatial_feat = True # Spatial features on or off
        self.hist_feat = True # Histogram features on or off
        self.hog_feat = True # HOG features on or off
        self.y_start_stop = [400, 665] # Min and max in y to search in slide_window()
        #self.y_start_stop = [300, 700] # Min and max in y to search in slide_window()
        self.vis_mode = {
                            "heat_view":False,
                            "window_view":False,
                            "map_view":False,
                            "detection_view":True,
                            "off_view":False
                        }
        self.frame_count = 0
        self.windows = []
        #self.windows += slide_window(x_start_stop=[0, 640], 
        #                    y_start_stop=[400,472], 
        #                    xy_window=(32,32), xy_overlap=(0.5, 0.5))
        #                    #xy_window=(48,48), xy_overlap=(0.85, 0.85))
        #self.windows += slide_window(x_start_stop=[600, 1280], 
        #                    y_start_stop=[334,664], 
        #                    xy_window=(64,64), xy_overlap=(0.85, 0.85))
        self.windows += slide_window(x_start_stop=[600, 1280], 
                            y_start_stop=[334,664], 
                            xy_window=(128,128), xy_overlap=(0.9, 0.9))
        #self.windows += slide_window(x_start_stop=[802, None], 
        #                y_start_stop=[292, 650], 
        #                xy_window=(64, 64), xy_overlap=(0.85, 0.85))                            
        #self.windows += slide_window(x_start_stop=[None, None], 
        #                    y_start_stop=[400,664], 
        #                    xy_window=(48,48), xy_overlap=(0.5, 0.5))
        self.hot_windows_saved = []
        self.use_saved_heat = True
        try:
            with open('svc_pickle.p','rb') as file_pi:
                self.svc, self.X_scaler = pickle.load(file_pi)
            print("Loaded pretrained model!")
        except:
            print("Could not find pretrained model")
            self.svc = None
            self.X_scaler = None
            self.train()

    def train(self):
        ## Read in cars and notcars
        cars = glob.glob('vehicles/**/*.png', recursive=True)
        notcars = glob.glob('non-vehicles/**/*.png', recursive=True)

        print("Number of cars:",len(cars))
        print("Number of notcars:",len(notcars))
        
        # Reduce the sample size because
        # The quiz evaluator times out after 13s of CPU time
        #sample_size = 100
        #cars = cars[0:sample_size]
        #notcars = notcars[0:sample_size]
        
        car_features = extract_features(cars, color_space=self.color_space, 
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat, notcar=False)
        notcar_features = extract_features(notcars, color_space=self.color_space, 
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat, notcar=True)
        
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)
            
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = self.X_scaler.transform(X_train)
        X_test = self.X_scaler.transform(X_test)
        
        print('Using:',self.orient,'orientations',self.pix_per_cell,
            'pixels per cell and', self.cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC3
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()

        with open('svc_pickle.p','wb') as file_pi:
            pickle.dump((self.svc, self.X_scaler), file_pi)

    def pipeline_function(self, image):
        print("#####Entering main pipeline for frame#####")
        #print(image)
        
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        #image = image.astype(np.float32)/255
        #sliding_windows_image = np.copy(image)
        #return_image = np.copy(image)

        ##roi_segments = np.round(np.linspace(self.y_start_stop[0],self.y_start_stop[1], num=3)).astype(int)
        #roi_segments = list(range(self.y_start_stop[0], self.y_start_stop[1], 
        #               (self.y_start_stop[1]-self.y_start_stop[0])//3))[::-1]
        ##print("rois",roi_segments)
        ##xy = [96, 96]
        #xy = [48, 48]


        ##roi_segments = [400,488,576,665]
        ##roi_segments = [665,576,488,400]
        #roi_segments = [665,473]
        #print("len roi segs", len(roi_segments))


        #for i in range(0,len(roi_segments)-1):
        #    print("i is",i)
        #    print("roi from",roi_segments[i+1],"to",roi_segments[i])
        #    windows = slide_window(image, x_start_stop=[None, None], 
        #                        y_start_stop=[roi_segments[i+1],roi_segments[i]], 
        #                        xy_window=(xy[0],xy[1]), xy_overlap=(0.5, 0.5))
        #    print("windows", windows)
        #    hot_windows = search_windows(image, windows, self.svc, self.X_scaler, color_space=self.color_space, 
        #                        spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
        #                        orient=self.orient, pix_per_cell=self.pix_per_cell, 
        #                        cell_per_block=self.cell_per_block, 
        #                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
        #                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)                       
    
        #    sliding_windows_image = draw_boxes(sliding_windows_image, windows, color=(0,1,0),thick=1)
        #    sliding_windows_image = draw_boxes(sliding_windows_image, hot_windows, color=(1,0,0),thick=1)
        #    hot_windows_all += hot_windows
        #    xy = [xy[0]//2, xy[1]//2]


        #windows += slide_window(image, x_start_stop=[None, None], 
        #                    y_start_stop=[self.y_start_stop[0],self.y_start_stop[1]//2], 
        #                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        #windows += slide_window(image, x_start_stop=[None, None], 
        #                    y_start_stop=[self.y_start_stop[0]//2,self.y_start_stop[1]],
        #                    xy_window=(48, 48), xy_overlap=(0.5, 0.5))
        #hot_windows = search_windows(image, windows, self.svc, self.X_scaler, color_space=self.color_space, 
        #                    spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
        #                    orient=self.orient, pix_per_cell=self.pix_per_cell, 
        #                    cell_per_block=self.cell_per_block, 
        #                    hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
        #                    hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        draw_image = np.copy(image)

            
        print("image size", image.shape[0], image.shape[1]) 
        #windows = []
        #windows += slide_window(image, x_start_stop=[0, 960], 
        #                    y_start_stop=[400,472], 
        #                    xy_window=(24,24), xy_overlap=(0.5, 0.5))
        #windows += slide_window(image, x_start_stop=[960, 1280], 
        #                    y_start_stop=[400,472], 
        #                    xy_window=(48,48), xy_overlap=(0.75, 0.75))
        #windows += slide_window(image, x_start_stop=[None, None], 
        #                    y_start_stop=[472,664], 
        #                    xy_window=(96,96), xy_overlap=(0.75, 0.75))

        #windows += slide_window(image, x_start_stop=[0, 640], 
        #                    y_start_stop=[400,472], 
        #                    xy_window=(48,48), xy_overlap=(0.85, 0.85))
        #windows += slide_window(image, x_start_stop=[640, 1280], 
        #                    y_start_stop=[400,664], 
        #                    xy_window=(96,96), xy_overlap=(0.85, 0.85))

        if self.use_saved_heat:
            hot_windows = self.hot_windows_saved[self.frame_count]
            self.frame_count += 1
        else:
            hot_windows = search_windows(image, self.windows, clf=self.svc, scaler=self.X_scaler, 
                                color_space=self.color_space, 
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)  
            self.hot_windows_saved.append(hot_windows)
        
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = add_heat(heat,hot_windows)
        #heat = apply_threshold(heat,1)
        #heatmap = np.clip(heat, 0, 255)
        self.heatmaps.append(heat)
        #heatmap_sum = sum(self.heatmaps)//len(self.heatmaps)
        heatmap_sum = sum(self.heatmaps)
        heatmap_sum = np.clip(heatmap_sum, 0, 255)
        #if len(self.heatmaps) < self.heat_len:
        #    heatmap_sum *= 0.3
        heatmap_sum = apply_threshold(heatmap_sum,self.heat_threshhold)
        labels = label(heatmap_sum)
        
        #sliding_windows_image = draw_boxes(image, windows, color=(0,1,0),thick=1)
        #sliding_windows_image = draw_boxes(image, hot_windows, color=(1,0,0),thick=1)
        #detection_image = draw_labeled_bboxes(image, labels)

        if self.vis_mode["window_view"]:
            draw_image = draw_boxes(draw_image, self.windows, color=(0,255,0),thick=1)
        if self.vis_mode["heat_view"]:
            print(draw_image.shape)
            draw_image = draw_boxes(draw_image, hot_windows, color=(255,0,0),thick=1)
        if self.vis_mode["map_view"]:
            empty_channel = np.zeros_like(heatmap_sum)
            for i in range(2): heatmap_sum = np.dstack((heatmap_sum, empty_channel))
            draw_image = heatmap_sum
            #output = np.copy(draw_image)
            #cv2.addWeighted(heatmap_sum.astype(np.int), 0.9, draw_image, 0.1, 0, output)
            #draw_image = output
        if self.vis_mode["detection_view"]:
            if labels[1]>0:
                draw_image, bboxes = draw_labeled_bboxes(draw_image, labels,self.bboxes_prev_frame)
                if bboxes: self.bboxes_prev_frame = bboxes
                self.prev_frame_labels = labels
            else:
                #draw_image = image
                print("hello")
                draw_image, bboxes = draw_labeled_bboxes(draw_image, self.prev_frame_labels,self.bboxes_prev_frame)
                if bboxes: self.bboxes_prev_frame = bboxes
        if self.vis_mode["off_view"]:
            draw_image = image
        return draw_image

def call_pipeline(image):
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                vid_processor_obj.vis_mode["heat_view"] ^= True
            elif event.key == pygame.K_m:
                vid_processor_obj.vis_mode["map_view"] ^= True
            elif event.key == pygame.K_w:
                vid_processor_obj.vis_mode["window_view"] ^= True
            elif event.key == pygame.K_d:
                vid_processor_obj.vis_mode["detection_view"] ^= True
            elif event.key == pygame.K_a:
                vid_processor_obj.vis_mode["all_view"] ^= True
            elif event.key == pygame.K_o:
                vid_processor_obj.vis_mode["off_view"] ^= True
    return vid_processor_obj.pipeline_function(image)

if __name__ == "__main__":
    pygame.init() 
    vid_processor_obj = MyVideoProcessor()
    #g_mtx, g_dist = calibrate_cameras()
    g_mtx = np.array([
             [  1.15396093e+03,   0.00000000e+00,   6.69705357e+02],
             [  0.00000000e+00,   1.14802496e+03,   3.85656234e+02],
             [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]
            ])
    g_dist = np.array([[ -2.41017956e-01,  -5.30721173e-02,  -1.15810355e-03,  -1.28318856e-04, 2.67125290e-02]])

    print("Camera calibration complete!")
    print("mtx:",str(g_mtx))
    print("dist:",str(g_dist))

    if vid_processor_obj.use_saved_heat:
        try:
            with open('heat_pickle.p','rb') as file_pi:
                vid_processor_obj.hot_windows_saved = pickle.load(file_pi)
            print("Loaded heat!")
            print(vid_processor_obj.hot_windows_saved)
        except:
            print("Could not find heat file")
            vid_processor_obj.use_saved_heat = False

    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML
    
    mode = "test" 
    #mode = "record" 
    if mode == "test":
        #clip1 = VideoFileClip("project_video.mp4").subclip(5,10)
        clip1 = VideoFileClip("project_video.mp4")
        output_clip = clip1.fl_image(call_pipeline)
        output_clip.preview()
    
    elif mode != "test":
        video_output = 'output.mp4'
        clip1 = VideoFileClip("project_video.mp4")
        output_clip = clip1.fl_image(call_pipeline)
        output_clip.write_videofile(video_output, audio=False)

    if not vid_processor_obj.use_saved_heat:
        with open('heat_pickle.p','wb') as file_pi:
            pickle.dump(vid_processor_obj.hot_windows_saved, file_pi)
            print("Pickled heat")

