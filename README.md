# Music Genre Prediction

The following repository consists of code to implement and test 2 methods of predicting music genre.
The dataset chosen for this project is sourced from this link: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Method1: This method uses images from the sourced website to predict the genre of the music. The method compares the performance of different architectures, training procedures and diffferent optimizers in performing classification. 
  - Three different model architectures were used.
      -** A feed forward Neural Network**
      -** A Convolution Neural Network**
      - A Convolution Neural Network with Batch Normalisation
  -  Two different optimisers were use.
      - Adam optimiser
      - RMS optimiser
  - 2 Different setting of training were use:
      - 50 epochs (No stopping criteria)
      - 100 epoch  (No stopping criteria)
   
Method2: This method uses audio files from the sourced website to predict the genre of the music. An **LSTM model architecture** was tested on two different audio set. One original clips, divided into 3 sec clips. Another dataset formed by implementing a **GAN**. The accuracy of LSTM model was test on the two dataset.

