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

- Developed `wav_to_tensor.py`

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

- Data looks good to train now

</observations>