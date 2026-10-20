# Accepted format for audio datasets

- For audio, the AutoML Engine expects a STRICT specific format of data
- You may upload a zipped folder ONLY
- The zip shall have the following structure
  - A CSV/TSV file with metadata (labels.csv or metadata.csv)
    - This file MUST have ATLEAST a column that has the filenames of the audio files AND a label
      - For example: filename will be clip1.wav and label will be speech
    - You may NOT have paths in the filename
      - For example: /Users/smukherjee/Downloads/alfie_automl_engine/data/clip1.wav is NOT ALLOWED.

  - A folder with the audio files (named `images` for consistency with the other engines)
    - The files here will be your audio clips (wav, mp3, flac, ogg, ...)

- An example format is as belows
- Note that you MUST zip this file. ONLY .zip is accepted.

```
➜  audio_subset git ✗ ls
images       labels.csv
➜  audio_subset git ✗ ls images
dog  siren
➜  audio_subset git ✗ ls images/dog
bark_001.wav bark_002.wav whimper_003.wav
➜  audio_subset git ✗ cat labels.csv | head -2
,filename,label
1,bark_001.wav,dog
2,siren_001.wav,siren
```
