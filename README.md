# mars_ejecta

Main scripts:
* cnn.py : Script to train a CNN to classify images of martian craters based on the presence of ejecta deposits. Calls functions from evaluation.py, model.py, and preprocessing.py
* run_unseen.py : script to predict the presence of ejecta in new images using the trained model from cnn.py
* texture_seg.py : functions and script to produce and classify feature vectors from texture descriptors (HOG, Gabor, ...)
* crater_finder_img.py : script to use the Hough transform and machine learning to find the center and radius of craters in images.
* saliency_map_tutorial.py

Libraries: 
* evaluation.py : functions to visualize the CNN
* model.py : functions to build and train the CNN
* preprocessing.py : functions to preprocess images for the CNN
