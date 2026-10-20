# Accepted format for text datasets

- For text, the AutoML Engine expects a STRICT specific format of data
- You may upload a zipped folder ONLY
- The zip shall have the following structure
  - A CSV/TSV file with metadata (labels.csv or metadata.csv) whose required
    columns depend on the task type:
    - `text_classification`: a text column (default name `text`, configurable
      via the `text_column` parameter) AND a `label` column
    - `question_answering`: `question`, `context`, `answer_start`, `answer_text`
    - `causal_lm`: `text`
    - `masked_lm`: `text`
    - `seq2seq_lm`: `input_text`, `target_text`
  - A folder with any media/asset files (named `images` for consistency with
    the other engines). For pure text datasets this folder may be empty, but
    it must exist inside the zip.

- Note that you MUST zip this file. ONLY .zip is accepted.

```
➜  imdb_text git ✗ ls
images       labels.csv
➜  imdb_text git ✗ cat labels.csv | head -2
,text,label
1,"A young girl tries to understand how she mysteriously gained the power to set things on fire with her mind.",horror
2,"Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",drama
```
