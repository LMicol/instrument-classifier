# Research Documentation

This file documents how I searched, my thought process and, the tools I've used to implement the solution.

### Personal Thoughts

Let's start by searching something.

---

###  Actions

- We start with a Google search: Audio classification  
- Found Google for Developers page  
- Looking into [Get started with audio classification](https://developers.google.com/learn/pathways/get-started-audio-classification)

---

### Personal Thoughts

WHAT? So we use computer vision???

---

### Research Oobservations

- Can use pre-trained model  
- Or train from scratch

---

### Personal Thoughts

Look into 1st idea

---

### Research Oobservations

- We can convert the audio into:  
  - Mel Spectrograms  
  - MFCCs (Mel-Frequency Cepstral Coefficients)

Spectrograms are 2D arrays and MFCCs are 1D arrays.  
1º Hypothesis: Spectrograms will have more information about the data and will take longer to train and predict.  
2º Hypothesis: MFCCs compress information into 1D, resulting in a loss of detail but decreasing the time to compute and potentially performing well enough compared to spectrograms.

---

###  Actions

- Developed tests/spectrograms.ipynb  
- Searched more about the topic  
- Found that we can use torch audio to do the processing  
- Developed tests/torchaudio.ipynb

---

### Research Oobservations

Torch audio simplifies the process, eliminating the need to generate images and convert them back to tensors, thereby preventing data loss.  
Tensors have different sizes similar to images; will need to resize them.

---

###  Actions

- Developed src/wav_to_tensor.py  
- Converted all wav to tensors.pt

---

### Research Oobservations

- There is no validation split in the dataset  
- I can re-structure the dataset to use a folder-based builder from Hugging Face

---

###  Actions

- Created a new script to load train data and generate validation split, while also creating the folder-based distribution

---

### Research Oobservations

Data looks good for training now. Should save to Hugging Face hub too.

---

###  Actions

- Created a simple load script src/upload_to_huggingface.ipynb using folder builder  
- Uploaded to Hugging Face [datasets](https://huggingface.co/datasets/Micol/musical-instruments-sound-dataset)  
- Started implementing src/models/1_model/train.ipynb

---

### Research Oobservations

During this implementation, I found that one audio file was corrupted; had to fix that before continuing.

---

### Personal Thoughts

Since I'll be using a MEL spectrogram and the audio can have different lengths, maybe I can convert the audio sample rate and other parameters to make it always the same size for the model.  
Also, I can try later doing the scrolling window approach.

---

###  Actions

- Re-uploaded the dataset  
- Finished implementing src/models/1_model/train.ipynb

---

### Personal Thoughts

While the first idea model trains, the validation accuracy appears to be around 50%, which is not good enough.  
My guesses are:  
  - The way I've implemented the preprocessing to pad/truncate the data is cropping the audio before the most recognizable part of the instruments, or the padding is leaving the spectrogram "too empty" for decent recognition.  
  - The duration of the slice may be too big; the average duration is 18 seconds, but I should've considered that this would dramatically affect short audios that are being padded.  
  - Or, maybe the neural network I've developed is too "simple", should try pre-trained models to see how that affects the results.

---

### Research Oobservations

Model finished training, evaluation can be seen at [ideas.md](ideas.md) for first idea results.

---

### Personal Thoughts

Next steps:  
  - Adjust preprocessing to only 2 seconds of audio  
  - Grab the 'most important' part of the audio, I'll check for the dB levels.

---

###  Actions

- Implemented src/models/2_model/train.ipynb  
- This crop is around the highest pitch of the audio; I've cropped one second from each side, adjusting in case it overflows the vector.

---

### Research Oobservations

Model two finished training, evaluation can be seen at [ideas.md](ideas.md) for second idea results.

---

### Personal Thoughts

Negligible margin of improvement, model did get 75% smaller though; let's try to use a pre-trained model as a base for the third idea.  
In the meantime, I'll implement the API provisioning and hosting.

---

###  Actions

- Implemented src/models/3_model/train.ipynb  
- Used an EfficientNet as a base model and modified the training loop a bit.

---

### Research Oobservations

Results got much better at validation, up to 60%.  
But when testing, got the same accuracy score as the second idea.  
Interestingly, there is only one major class getting mislabeled; most violin sounds are being labeled as piano. For the other classes, things are much better.

---

### Personal Thoughts

Two last things I want to test before deploying the API are testing ResNet and MFCCs.  
ResNet will be just a change on the third model architecture, while for the MFCCs I will need to rework the pipeline.

---

###  Actions

- Implemented src/models/4_model/train.ipynb  
- Used a ResNet152 as a base model and increased the sampling rate.

---

### Research Oobservations

Model is going crazy during training this time, accuracy went from 0.60 -> 0.26 -> 0.65 -> 0.30.  
This will take a bit longer too since ResNet152 is a bigger model.

---

###  Actions

- Implemented src/models/5_model/train.ipynb  
- Made a simple neural network like the 2nd model, used the latest training loop, and modified data processing to use MFCCs.

---

### Research Oobservations

Model 4 train ended, complete garbage.  
Everything is being classified as Violin sound.

---

### Personal Thoughts

This is so bad, I'm not even saving the weights to this repository.

---

### Research Oobservations

Model 5 train ended.  
Model converged around the 20th epoch.  
Solid increase in performance.  
5th model weights less than 1MB.  

1st Hypothesis => Wrong.  
2nd Hypothesis => Wrong.  
Spectrograms and MFCCs took almost the same time to train.  
MFCCs did perform better and converged much faster.

---

### Personal Thoughts

60% accuracy is acceptable.  
Time to deploy the fifth model.

---

###  Actions

- Developed src/server  
- Developed src/web  
- Developed docker-compose

---

### Research Oobservations

Now we have a simple web page to test the model, you can load a file or use the microphone you select to stream the audio and see the results of the deployed model.

---

### Personal Thoughts

I've been playing around with the model on the web page; I'm not really liking the results.  
For piano/keyboard, it performs really well.  
For my electric guitar, it performs decently but mislabels a couple of times. When I'm using the acoustic guitar, it's awful.

Since all the framework is already done, I'll give it one more try to improve this model.

---

###  Actions

- While searching for possibilities, I've found [wav2vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2), seems promising.  
- Adapting train notebook to use wav2vec2.  
- Trained for a couple of epochs, performed super bad, scraping idea.

---

### Personal Thoughts

Okay, I've played a bit more with the spectrograms, maybe cutting the audio was a bad idea.  
I'll try to extract the whole data from the audio, not just a cut as it is right now, and resize the Mel Spectrograms itself.  
And use that back again into the CNN as an image, just resizing the input image when necessary.

---

###  Actions

- Developed src/models/6_model/train.ipynb

---

### Research Oobservations

Really good accuracy when training, over 80%. On the test set performed <50% lol.  
Tried a couple of different things this time too, like reducing batch size and adding MaxPool layers.  
Also, increased some of the MEL parameters.  

Overall, the models seem to plateau close to the same mark, for sure there is a way to preprocess the audios that would make the final accuracy much better.

---

### Personal Thoughts

Well, for now, that's it, maybe I come back to this problem another day =)