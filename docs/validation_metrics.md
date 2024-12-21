# Metrics
## Training
Categorical Cross Entropy is used as a loss metric during training

## Validation

### Pitchwise f1 score
Metric as defined in [MuTE](https://github.com/matangover/mute/tree/main)

Evaluate similarity between target and reference using the F1 score -- precision and recall.
What proportion of notes in the reference were correctly recalled in the target (recall), and
what proportion of the notes in the target are in fact correct (precision). This is done per
pianoroll time-slice. We treat it as a multiclass/multilabel classification problem for each
time step, and then use the F1 score for each time step, and average the F1 scores.

First we normalize the pitch values to one octave, then perform the metric calculation.
