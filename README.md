# Face_Detection
Objective Comparing four different models of face detection 
1. Haar Cascade 2. DNN Caffe Model 3. HoG Model 4. DNN YOLO Model

Description of Models
1. Haar Cascade:
   A classic and lightweight algorithm for object detection, including faces, based on pre-defined features and cascading classifiers
   Pros: •	Simple and fast. •	Low computational requirements.
   Cons: •	Limited accuracy, especially in challenging conditions (e.g., occlusions, varying lighting conditions). •	Doesn’t work on non-frontal images. •	Prone to false positives.
2. DNN Caffe Model: Introduction: A deep neural network model for face detection trained using the Caffe framework, offering high accuracy but requiring significant computational resources.
   Pros: •	High accuracy, especially with large and diverse datasets. •	Can handle challenging conditions better than traditional methods.
   Cons: •	Higher computational requirements compared to Haar Cascade. •	Requires a trained model and a more complex setup.
3. HoG Model: Introduction: Utilizes Histogram of Oriented Gradients to detect object boundaries, including faces, with moderate accuracy and lower computational requirements compared to deep learning models.
   Pros: •	Moderate accuracy with relatively lower computational requirements compared to deep learning models. •	Works very well for frontal and slightly non-frontal faces •	Lightweight model as compared to the
   other three.
   Cons: •	The major drawback is that it does not detect small faces, as it is trained for a minimum face size of 80×80. Thus, you need to ensure that the face size is more than that in your application. You can,
   however, train your own face detector for smaller-sized faces. •	Less accurate than deep learning-based approaches, especially with complex backgrounds. •	Slower than Haar Cascade for real-time applications.
4. DNN YOLO Model: Introduction: Employs the You Only Look Once (YOLO) architecture for efficient real-time face detection by simultaneously predicting bounding boxes and class probabilities, suitable for
   applications requiring speed and accuracy
   Pros: High accuracy and robustness, like the DNN Caffe Model. Faster inference compared to some other deep learning models due to its single-pass architecture.
   Cons: Higher computational requirements compared to traditional methods like Haar Cascade and HoG. May require more training data and fine-tuning for optimal performance

We check for the following things while comparing the models over a small dataset.
Accuracy: Compare the accuracy of each model based on benchmarks and real-world performance.
Speed: Discuss the inference speed of each model, considering real-time applications.
Robustness: Evaluate how well each model performs under challenging conditions like varying lighting, occlusions, and scale variations.

Findings and Discussion:

Accuracy:
It is detected by upscaling the images for HoG by a factor of 2 to ensure better results. Haar is the most inaccurate of the lot. Both the CNN methods do well in terms of accuracy and HoG is also quite accurate when upscaled.

Speed: 
HoG takes a longer time due to the upscaling required. Both the CNN methods outperform the other two in terms of speed. CNN-YOLO method however is faster as YOLO performs object detection in a single pass through the neural network compared to traditional object detection algorithms that use sliding windows or region-based approaches.
Test Under Varying Conditions:

   1. Scale:
  The lib-based method i.e. HoG can detect faces of size up to ~(70×70) after which they fail to detect. As we discussed earlier, I think this is the major drawback of Dlib-based methods. Since it is impossible     to know the size of the face beforehand in most cases, we can get rid of this problem by upscaling the image, but then the speed advantage of dlib as compared to OpenCV-DNN goes away.
  Insert images of HoG with varying scales.

  2: Non-Frontal Face
  As expected, Haar based detector fails. The HoG-based detector does detect faces for left or right-looking faces (since it was trained on them ) but is not as accurate as the DNN-based detectors of OpenCV and     Dlib.
  
  3. Occlusion
  The DNN-Caffe methods outperforms the others, even though it is slightly inaccurate. This is mainly because the CNN features are much more robust than HoG or Haar features.
  In general, the CNN Methods are more robust, providing better results in all conditions. The robustness of HoG depends on the scale of the images. It performs well with upscaled images but fails when faces are smaller.

Conclusion

General Case

In most applications, we won’t know the face size in the image beforehand. Thus, it is better to use DNN-Caffe method as it is pretty-fast and very accurate, even for small sized faces. It also detects faces at various angles. We recommend using DNN-Caffe in most cases. However if your computer can handle high GPU Computations then we recommend DNN-YOLO.

For medium to large image sizes

Dlib HoG is the fastest method on the CPU. But it does not detect small sized faces ( < 70×70 ). So, if you know that your application will not be dealing with very small sized faces then HoG based Face detector is a better option.

High-resolution images

Since feeding high-resolution images is not possible with these algorithms ( for computation speed ), HoG detectors might fail when you scale down the image. On the other hand, the DNN-YOLO method can be used for these since it detects small faces.

In essence

Haar Cascade can be used in scenarios that require real-time applications with limited computational resources.
DNN-Caffe can be used in applications where accuracy is critical and computational resources are sufficient.
HoG Model might be preferable in applications where accuracy and speed is important but computational resources are limited.
DNN YOLO Model excels in applications requiring both accuracy and speed and where computational resources are available.

