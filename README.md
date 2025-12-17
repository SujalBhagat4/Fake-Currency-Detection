\# Fake Currency Detection System using Deep Learning (CNN)



\## Project Overview



This project is an AI-powered Fake Currency Detection System that identifies whether an Indian currency note is Real or Fake using Deep Learning (Convolutional Neural Networks). The system is deployed as a Flask web application where users can upload an image of a currency note and receive instant results along with a detailed PDF authentication report.



The project focuses on accuracy, robustness, and real-world usability by applying multi-angle image analysis and majority voting.



---



\## Key Objectives



\* Detect fake Indian currency notes using image-based analysis

\* Apply deep learning techniques for feature extraction and classification

\* Provide a simple and user-friendly web interface

\* Generate professional PDF reports for each prediction



---



\## Technology Stack



Frontend:



\* HTML5

\* CSS3

\* Bootstrap

\* JavaScript



Backend:



\* Python

\* Flask

\* Flask-CORS



Machine Learning / AI:



\* TensorFlow

\* Keras

\* Convolutional Neural Network (CNN)

\* MobileNetV2 (Transfer Learning)

\* OpenCV

\* NumPy



Reporting:



\* ReportLab (PDF generation)



---



\## System Architecture



1\. User uploads a currency note image via the web interface

2\. Image is sent to the Flask backend

3\. Image preprocessing (resize, normalization, RGB conversion)

4\. CNN model predicts authenticity score

5\. Image is tested at four rotations (0°, 90°, 180°, 270°)

6\. Majority voting decides the final result

7\. PDF report is generated

8\. Result is displayed to the user



---



\## Deep Learning Model Details



\* Model Type: Convolutional Neural Network (CNN)

\* Base Model: MobileNetV2 (pre-trained on ImageNet)

\* Input Size: 224 x 224 x 3

\* Output: Binary classification (Real / Fake)

\* Loss Function: Binary Crossentropy

\* Optimizer: Adam

\* Activation Functions: ReLU (hidden layers), Sigmoid (output layer)



Why Transfer Learning:



\* Faster training time

\* Better accuracy with limited data

\* Efficient feature extraction from complex images



---



\## Multi-Angle Prediction Logic



To improve prediction reliability, the system:



\* Rotates the uploaded image at 0°, 90°, 180°, and 270°

\* Performs prediction on each rotated image

\* Uses majority voting to determine the final classification



This approach improves robustness against image orientation issues.



---



\## PDF Report Features



Each prediction generates a downloadable PDF containing:



\* Date and time of analysis

\* Final authenticity result

\* Confidence score

\* Rotation-wise prediction scores

\* Vote summary (Real vs Fake)

\* Uploaded currency image



---



\## Project Structure



```

Fake-Currency-Detection/

│

├── app.py                  # Flask backend

├── model.keras             # Trained CNN model

├── index.html              # Frontend user interface

├── uploads/                # Uploaded images

├── reports/                # Generated PDF reports

├── fake\_currency\_cnn.ipynb # Model training notebook

├── requirements.txt        # Project dependencies

└── README.md               # Project documentation

```



---



\## How to Run the Project



1\. Install dependencies



```bash

pip install -r requirements.txt

```



2\. Run the Flask application



```bash

python app.py

```



3\. Open the application in a browser



```

http://localhost:5000

```



---



\## Sample Output



\* Prediction: Fake Currency

\* Confidence: 92 percent

\* Votes: Fake (3) | Real (1)

\* Output Report: Auto-generated PDF



---



\## Future Enhancements



\* Support for multiple currency denominations

\* Model explainability using Grad-CAM

\* Mobile application integration

\* Cloud deployment (AWS, Azure)

\* Larger and more diverse dataset for improved accuracy



---



\## Author



Sujal

B.Tech in Artificial Intelligence



---



\## License



This project is developed for educational and academic purposes only.



---



If you find this project useful, consider starring the repository on GitHub.



