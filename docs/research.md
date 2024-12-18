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

- Started implementing `src/1_model/train.ipynb`

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

- Finish implementing `src/1_model/train.ipynb`

</actions>

<tought>

While the first idea model trains, the validation accuracy appears to be around 50%, which is not good enough.
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

<actions>

- Implemented `src/2_model/train.ipynb`

- This all crops are around the highest pitch of the audio, I've croped one second to each side, adjusting in case it overflows the vector.

</actions>

<observations>

Model two finished traning, evaluation can be seen at [ideas.md](ideas.md) second idea results.

</observations>

<tought>

Negligible margin of improvement, model did get 75% smaller tho, let's try to use a pre-trained as a base for the third idea.

In the mean time I'll implement the API provisioning and hosting.

</tought>

<actions>

- Implemented `src/3_model/train.ipynb`

- Used an EfficientNet as base model and modified the traning loop a bit

</actions>

<observations>

Results got much better at validation up to 60%.

But when testing got the same accuracy score as the second idea.

Interestly, there is only one major class getting misslabeled, most of the violin sounds are getting piano labeled, for the other classes, things are much better.

</observations>


<tought>

Two last thing I want test before deploying the api is testing ResNet and MFCCs.

ResNet will be just a change on the thrird model architecture, while for the MFCCs I will need to rework the pipeline.

</tought>
