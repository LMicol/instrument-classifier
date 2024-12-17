This file documents how I searched, my thought process and, how I found the tools to implement the solution

<tought>Let's start by searching something</tought>

<actions>

- We start with a Google search: `Audio classification`

- Found Google for Developers page

- Looking into [Get started with audio classification](https://developers.google.com/learn/pathways/get-started-audio-classification)

</actions>

<tought>WHAT? So we use computer vision???</tought>

<observations>

- Can use pre-trained model

- Or train from scratch

</observations>

<tought>Look into 1st idea</tought>

<observations>

- We can convert the audio into:
    - Mel Spectrograms
    - MFCCs (Mel-Frequency Cepstral Coefficients)

Spectograms are 2D arrays and MFCCs are 1D arrays.
1º Hypothesis: Spectograms will have more information about the data and will take longer to train and predict.
2º Hypothesis: MFCCs compress information into the 1D resulting a loss of detail but decrease the time to compute and can perform good enought when compared to the spectrograms. 

</observations>

<actions>

- Developed `tests/spectrograms.ipynb`

- Searched more about the topic

- Found that can use torch audio to do the processing

- Developed `tests/torchaudio.ipynb`

</actions>

<observations>

Torch audio makes things much simpler, no need to generate images and convert it back to tensors, skiped one processed and prevented data loss.

Tensors have different size like the images, will need to resize.

</observations>

<actions>

- Developed `src/wav_to_tensor.py`

- Converted all wav to tensors.pt

</actions>

<observations>

- There is no validation split on the dataset

- I can re-structure the dataset to use folder-based builder from hugging face

</observations>

<actions>

- Created a new script to load train data and generate validation split, while also creating the folder-based distribution

</actions>

<observations>

Data looks good to train now. Should save to hugging face hub too.

</observations>

<actions>

- Created simple load script `src/upload_to_huggingface.ipynb` using folder builder

- Uploaded to Hugging Face [datasets](https://huggingface.co/datasets/Micol/musical-instruments-sound-dataset)

- Started implementing `src/first_model/train.ipynb`

</actions>

<observations>

During this implementation I found that one audio file was corrupted, had to fix that before continuing.

</observations>

<tought>

Since I'll be using a MEL spectrogram and the audio can have different lengths, maybe I can convert the audio sample rate and other parameters to make it always the same size for the model.

Also, I can try later doing the scrolling window approach.

</tought>

<actions>

- Re-uploaded the dataset

- Finish implementing `src/first_model/train.ipynb`

</actions>

<tought>

While the first idea model trains, the validation accuracy appears to be around 60%, which is not good enough.
My guesses are:
    - The way I've implemented the pre processing to pad/truncate the data is croping the audio before the most recognizable part of the instruments, or the pading is leaving the spectrogram "too empty" for a descent recognition.
    - The duration of the slice may be too big, the average duration is 18 seconds, but I should've consider that this would dramatically afect short audios that are being padded.
    - Or, maybe the neural network I've developed is too "simple", should try pre-trained models to see how that afect the results.

</tought>

<observations>

Model finished traning, evaluation can be seen at [ideas.md](ideas.md) first idea results.

</observations>

<tought>

Next steps:
    - Adjust preprocessing to only 2 seconds of audio
    - Grab the 'most important' part of the audio, I'll check for the dB levels.

</tought>