# CloudGaze

Landsat Sky Image based Determination of Cloud Motion to Predict Solar Obscuration  

## Project Overview 
The detection and prediction of cloud motion from ground based visible light images are critical for optimizing solar energy generation and improving solar facility yield predictions. Moreover, understanding cloud dynamics provides valuable insights for climate modeling.
**CloudGaze** is a cloud motion detection and prediction system designed to analyze the movement of clouds using LandSat images. The project applies image segmentation techniques, motion detection algorithms, and predictive modeling to assess potential sun coverage and forecast cloud movement.

## Features  
- **Cloud and Sun Detection:** Binary segmentation of clouds and sun regions from Landsat sky images.  
- **Optical Flow Analysis:** Lucas-Kanade algorithm to track cloud motion vectors.  
- **Centroid Calculation:** Identifies largest cloud and sun positions for spatial analysis.  
- **Predictive Modeling:** Estimates cloud motion and sun coverage probabilities using a Kalman filter.   
- **Real-Time Processing:** Periodic image capture using an RPi camera module for dynamic cloud tracking.  
 

## Project Structure  
```plaintext
CloudGaze/
├── data/
│   ├── filtered/          # Original LandSat images
│   ├── followed/
|   ├── movement/
│   ├── predicted/
│   ├── followed/
│   ├── presentation/
│   ├── processed/
│   ├── saved/
│   ├── segmented/
│   └── segmented/thresholded/
├── src/      # Cloud motion detection and prediction
│   ├── version1.py/   
│   └── kalmanfilter.py/
├── data/
│   ├── network.py/               # Helper functions and utilities
├── models/            # Pre-trained and custom segmentation models
└── README.md          # Project documentation

