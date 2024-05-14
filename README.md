# Musical_Instrument_Audio_Classification

First of all, clone this GitHub repository in your local by running

```
git clone https://github.com/fernandocarballeda/Musical_Instrument_Audio_Classification.git
```

in your terminal, while you are in the directory you want the repository to get created. Go inside the ``Musical_Instrument_Audio_Classification`` folder and execute the following commands to create a conda environment and install the necessary libraries:

```
conda create -n miac python=3.7

activate miac

pip install -r requirements.txt
```

Now, you can open Jupyter or Visual Studio Code to start looking and using the content.

## Audio files

``wavfiles`` folder contains audios in .wav format of different musical instruments.

<img width="225" alt="image" src="https://github.com/fernandocarballeda/Musical_Instrument_Audio_Classification/assets/144034714/a4b5db6e-8148-4297-b9e9-a80ada99ef7f">

If we want to add more data to this folder in order to train with more data or include more instruments in the classificator we have to download from the Internet the audio and include it in this folder. If the audio is in .mp3 format, using FFmpeg converter, you can change its file type to .wav using this command in the terminal:

```
C:\ffmpeg\bin\ffmpeg -i audio_name.mp3 audio_name.wav
```

Here is some useful information on how to install this converter: https://phoenixnap.com/kb/ffmpeg-windows

## Data cleaning

``clean.py`` script provides a set of tools for processing and cleaning the audio data.

It starts by splitting audios in ``wavfiles`` folder into smaller segments (1.0 seconds by default) and deleting those with no sound (a threshold can be specified to define what an audio with no sound is). Afterwards, the valid segments of audio are stored in ``clean_folder`` divided again by musical instruments.

To execute this script, run the next command in the terminal:

```
python clean.py
```

you can specify different params like

``--delta_time`` to choose the time in seconds of the generated audios.

## Models

``models.py`` contains the definition of the models we will train to try to perform a good classification of the audios. It contains some models from scratch and other pretrained ones. Feel free to include new models in order to get a better accuracy in the classification.

## Training

``train.py`` is in charge of training the different models defined in the ``models.py`` script. It performs a train_test_split with the data contained in ``clean_folder``.

By changing the ``model_type`` parameter we can choose the model we want to train, like this:

```
python train.py --model_type lstm
```

Once the model is trained, it is stored in ``models_folder``.

<img width="221" alt="image" src="https://github.com/fernandocarballeda/Musical_Instrument_Audio_Classification/assets/144034714/2716915b-3f40-4414-bd4f-c03e4d94baf4">


## Logs (accuracy)

When a model is trained using the ``train.py`` script, a new file is generated in this folder, with information about the training and validation accuracy and training and validation loss for each epoch.

<img width="497" alt="image" src="https://github.com/fernandocarballeda/Musical_Instrument_Audio_Classification/assets/144034714/169d8940-9d2b-4797-97a5-864e8e61fd8e">

## Prediction

In ``predict_folder`` we will deposit the audio that we want to predict (**Remember:** only one audio at a time). When executing the ``predict.py`` script the process will start. A new folder called ``predict_dir_cleaned`` will be created, containing the one-second duration segments of the input audio. A function will be applied to this folder to delete those segments with no audio (something like this can happen). Afterwards, you will see in the console the prediction for each segment, the Mean Prediction Class (the class with more average probability) and the Global Prediction (most predicted class for all the segments).

**Important:** Once you have predicted an audio, you must delete the ``predict_dir_cleaned`` folder to avoid errors when performing the next prediction.

## Validation data

This folders will contain the audios (in .mp3 and .wav format respectively) that we can use to test the classificator. If you want to try how does it work with your own audios, download them and place them here.
