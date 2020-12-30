# Sound_Based_Emotion_Analyzer
Joheun Kang 

## Data

1.RAVDESS <br>
2.TESS <br>

RAVDESS data is in the __raw_data_1__ and the TESS data is in the __raw_data_2__.
You can download the two folders from my Google Drive using below link.

https://drive.google.com/drive/folders/1aD1XpONg9xFdUCvIELXqHxgmPIu3FlET?usp=sharing


## Program & Pre-Trained model

In the folder named "saved_models", there are <br>

1. emotion_detection.py<br>
2. model_new.json<br>
3. new.h5<br>

model_new.json and new.h5 are the pre-trained model and the new.h5 is its weights. 


## Run Program

To run the program, please download "saved_models" file ONLY. 
Then, go to the "saved_models" folder and run 

```bash
$ python emotion_detection.py
```

Then, the program will run, and the recored file will be saved.


## Check Training Process
If you want to check the training process, please open "emotion_analyzer_model_training.ipynb" and see how the model is trained. 
