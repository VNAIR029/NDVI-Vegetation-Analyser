# Description
This is a simple NDVI Vegetation Analyser that I made using Python and Streamlit. While I made some changes to the code, 
most of the credit has to go to two very helpful tutorials that I had followed. The first one being "Image classification with Python and Scikit learn | Computer vision tutorial" by Computer vision engineer on Youtube for the main body of the analyser 
and "Satellite Imagery Analysis using Python" by Gaurav Dutta on Kaggle.

The model runs using Streamlit (as mentioned earlier) and you can upload NDVI images unto it (but it must be in an RGB format, like a screenshot of a NDVI image). 
What it then does is that it classifes the image into three types by taking the average NDVI value; Low drought risk (Healthy vegetation), Moderate drought risk (Watch conditions), High drought risk (Action needed).
Additonally, it can classify these images in two diffrent modes, by simply doing an RBG analysis of the NDVI image or by creating a heatmap.

Furthermore, it can analyse entire directories as well as single images.

I'll be making a Medium article soon about my process in making this as well as the purpose behind it.
