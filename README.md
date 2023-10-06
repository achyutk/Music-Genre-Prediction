# Music Genre Prediciton
![img](https://github.com/achyutk/Music-Genre-Prediction/assets/73283117/24ad3028-2d2f-4b9c-bfd7-5063980a9528)


The following repository consists of code to predicting music genre using images of MEL Spectograms. The dataset chosen for this project is sourced from this link: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

This method uses images from the sourced website to predict the genre of the music. The method compares the performance of different architectures, training procedures and diffferent optimizers in performing classification.

  - Three different model architectures were used.
      - **A feed forward Neural Network**
      - **A Convolution Neural Network**
      - A Convolution Neural Network with Batch Normalisation
  -  Two different optimisers were use.
      - Adam optimiser
      - RMS optimiser
  - 2 Different setting of training were use:
      - 50 epochs (No stopping criteria)
      - 100 epoch  (No stopping criteria)



# Installations

Clone repo and install the following libraries:

> pip install torch torchvision torchaudio <br>
> pip install sklearn


Download the [images_original Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and paste in the *data* folder of this repository

train.py file will execute and save the best model achieved which is the third CNN optimised using AdamOptimiser. <br>
demo.ipynb  demonstrates exectution of all the models that were tested.

# Results

Accuracy achieved on the validation dataset formed from the dataset:

| Model  | AdamOptimiser_Val accuracy    | RMSOptimiser_Val accuracy|
 :---: | :---: | :---: |
| FNN:50 epochs | 46.46%   | 8.08%   |
| FNN:100 epochs | 43.43%   | 8.08%   |
| CNN-1: 50 epochs| 8.08%   | 8.08%   |
| CNN-1: 50 epochs| 8.08%   | 8.08%   |
| CNN-2: 50 epochs| 68.68%   | 16.16%   |
| CNN-2: 50 epochs| 66.66%   | 53.53%   |

# Further Reading

The report of this project can be found on the following link:
https://drive.google.com/file/d/11vmE0u58IIIVIq__QkDbb_oKJxXXR3U7/view?usp=sharing

Update version of a different method with GAN-LSTM implementation can be found in this [repository](https://github.com/achyutk/Music-Genre-Prediction--v2)
