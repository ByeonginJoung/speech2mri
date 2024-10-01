# Speech to MRI

This repository provides a source code to convert audio speech to 2D real-time MRI image.

```
python=3.10.14

pytorch==1.13.1+cu117
librosa==0.10.2.post1
matplotlib==3.6.2
numpy==1.23.5
moviepy==1.0.3
scikit-learn==1.9.3
pydub
soundfile
opencv-python
tqdm
pytorch_msssim
```

I used the [USC-span dataset](https://sail.usc.edu/span/75speakers/), 75 speakers to train the model.

The detail to train the model and demo is introduced in bash file.