# Perceptron-based-Image-classification
Designing a Perceptron-based classification algorithm that can recognize scanned images of the 10 digits (0 to 9)

The "Dataset" zip file contains three folders: “Train”, “Validation” and “Test”.
The “Train” folder contains 240 images for each digit, while each of the “Validation” and “Test” folders contain 20 images for each digit. 

Requirements:
1) You need to train the classifiers using each of the following values for the learning rate η = 1, 10-1, 10-2, 10-3, 10-4, 10-5, 10-6, 10-7, 10-8, 10-9. For all Perceptrons, use an initial weight vector that has 1 as the first component (w1) and the rest are zeros.

2) After the classifiers are trained, test each classifier using the images given in the “Test” folder. The folder also contains a text file named “Test Labels.txt” which include the labels of the 200 images in order, then create a confusion matrix for each value of η showing the number of images of the “Test” folder of each digit that were classified to belong to different digits (For example: Number of images of 0 that were classified as 0, 1, 2, …, 9, and so on for other digits).

3) Use the data in the “Validation” folder to find the value of η that achieves the best accuracy for each digit classifier. Use the best classifier of each digit to classify the data in the “Test” folder. The “Validation” folder also contains a text file named “Validation Labels.txt” which include the labels of the 200 images in order.

Important Notes:
• Do not use Python built-in functions for the Perceptron. You have to implement your own version of all needed functions. You are only allowed to use functions that load images into Python.
