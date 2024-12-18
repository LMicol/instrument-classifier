<first>
Objective:
    - Classify audio into [Guitar, Drum, Violin, Piano]

Method:
    - Create spectogram images from audio files
    - Define simple CNN as baseline
    - Train and evaluate

Metrics:
    - Cross-Entropy
    - Accuracy
    - Confusion matrix

Results:
    - Final Acc = 52.5%

<img title="Confusion Matrix for first idea" src="../images/1_idea.png">

</first>

<second>
Objective:
    - Classify audio into [Guitar, Drum, Violin, Piano]

Method:
    - Find the loudest sound on the audio and grab 1s to each direction.

Metrics:
    - Same metrics

Results:
    - Final Acc = 53.75%

<img title="Confusion Matrix for second idea" src="../images/2_idea.png">
</second>

<third>
Objective:
    - Classify audio into [Guitar, Drum, Violin, Piano]

Method:
    - Find the loudest sound on the audio and grab 1s to each direction.
    - Use a pre-trained model as a base model (Efficient Net).

Metrics:
    - Same metrics

Results:
    - Final Acc = 53.75%

<img title="Confusion Matrix for third idea" src="../images/3_idea.png">

</third>

<fourth>
Objective:
    - Classify audio into [Guitar, Drum, Violin, Piano]

Method:
    - Find the loudest sound on the audio and grab 1s to each direction.
    - Use a pre-trained model as a base model (ResNet).
    - Increase Sample Rate

Metrics:
    - Same metrics

Results:
    - Final Acc = 17.50%

<img title="Confusion Matrix for fourth idea" src="../images/4_idea.png">

</fourth>

<fifth>
Objective:
    - Classify audio into [Guitar, Drum, Violin, Piano]

Method:
    - Find the loudest sound on the audio and grab 1s to each direction.
    - Compute the MFCCs of the sound.
    - Train on a basic CNN.

Metrics:
    - Same metrics

Results:
    - Final Acc = 61.25%

<img title="Confusion Matrix for fourth idea" src="../images/5_idea.png">

</fifth>