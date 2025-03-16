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
  
 ## 1. Install Miniconda and Requirements

### Download CloudGaze Repository
```bash
git clone https://github.com/weijithw2022/CloudGaze.git
cd CloudGaze
```

### Install to "ml_base" Virtual Environment
```bash
conda env create -f environment.yaml
conda activate ml_base
```
 ## 2. Download the dataset
 
[Google Drive Link](https://drive.google.com/file/d/1M7_Cnm79XJ7fVagPmRALkzF5Hbb5ypc-/view?usp=drive_link)

 ## 3. Run the code
 ```bash
python src/version1.py
```


## Project Structure  
```plaintext
CloudGaze/
├── data/
│   ├── filtered/          
│   ├── followed/
│   ├── movement/
│   ├── predicted/
│   ├── presentation/
│   ├── processed/
│   ├── saved/
│   ├── segmented/
│   │   └── thresholded/
├── src/     
│   ├── version1.py   
│   ├── kalmanfilter.py
│   ├── models/
│   │   ├── network.py
│   │   ├── loss.py           
│   ├── temp/
│   │   ├── avgvector.py
│   │   ├── contours.py
│   │   ├── filterbrightmask.py
│   │   ├── findcentroid.py
│   │   ├── imagecap.py
│   │   ├── largestcontour.py
│   │   ├── lucasKanademodel.py
│   │   ├── lucasKanadethresholded.py                     
├── models/            
└── README.md


```

