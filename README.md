# [Supervised Learning : Learning Hierarchical Features for Predicting Multiclass X-Ray Images using Convolutional Neural Network Model Variations](https://johnpaulinepineda.github.io/Portfolio_Project_44/)

[![](https://img.shields.io/badge/Python-black?logo=Python)](#) [![](https://img.shields.io/badge/Jupyter-black?logo=Jupyter)](#)

The integration of artificial intelligence (AI) into healthcare has emerged as a transformative force revolutionizing diagnostics and treatment. The urgency of the COVID-19 pandemic has underscored the critical need for rapid and accurate diagnostic tools. One such innovation that holds immense promise is the development of AI prediction models for classifying medical images in respiratory health. This [case study](https://johnpaulinepineda.github.io/Portfolio_Project_44/) aims to develop multiple convolutional neural network (CNN) classification models that could automatically learn hierarchical features directly from raw pixel data of x-ray images (categorized as Normal, Viral Pneumonia, and COVID-19), while delivering accurate predictions when applied to new unseen data. Data quality assessment was conducted on the initial dataset to identify and remove cases noted with irregularities, in addition to the subsequent preprocessing operations to improve generalization and reduce sensitivity to variations most suitable for the downstream analysis. Multiple CNN models were developed with various combinations of regularization techniques namely, Dropout for preventing overfitting by randomly dropping out neurons during training, and Batch Normalization for standardizing the input of each layer to stabilize and accelerate training. CNN With No Regularization, CNN With Dropout Regularization, CNN With Batch Normalization Regularization, and CNN With Dropout and Batch Normalization Regularization were formulated to discover hierarchical and spatial representations for image category prediction. Epoch training was optimized through internal resampling validation using Split-Sample Holdout with F1 Score used as the primary performance metric among Precision and Recall. All candidate models were compared based on internal and external validation performance. Post-hoc exploration of the model results involved Convolutional Layer Filter Visualization and Gradient Class Activation Mapping methods to highlight both low- and high-level features from the image objects that lead to the activation of the different image categories. These results helped provide insights into the important hierarchical and spatial representations for image category differentiation and model prediction.

<img src="images/CaseStudy5_Summary_0.png?raw=true"/>

<img src="images/CaseStudy5_Summary_1.png?raw=true"/>

<img src="images/CaseStudy5_Summary_2.png?raw=true"/>

<img src="images/CaseStudy5_Summary_3.png?raw=true"/>

<img src="images/CaseStudy5_Summary_4.png?raw=true"/>

<img src="images/CaseStudy5_Summary_5.png?raw=true"/>

<img src="images/CaseStudy5_Summary_6.png?raw=true"/>
