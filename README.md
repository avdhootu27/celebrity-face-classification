# celebrity-face-classification


This is a machine learning project to classify image of 5 players. You just need to upload or drag & drop an image and web app will classify it and will tell with what probability that image matches with those 5 players.
For now the website is not hosted anywhere but the code of frontend and backend is present in UI and server folder respectively.

The model folder has the main python code along with trained model and dataset. The model is trained using SVM and [opencv-haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades) are used to detect eyes and face of a person in an image.
