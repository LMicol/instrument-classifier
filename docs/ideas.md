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
</second>

<third>
Objective:
    - Classify audio into [Guitar, Drum, Violin, Piano]

Method:
    - Find the loudest sound on the audio and grab 1s to each direction.
    - Use a pre-trained model as a base model.

Metrics:
    - Same metrics

Results:
    - TBD
</third>