# Accepted format for vision datasets

- For vision, the AutoML Engine expects a STRICT specific format of data
- You may upload a zipped folder ONLY
- The zip shall have the following structure
  - A CSV/TSV file with metadata
    - This file MUST have ATLEAST a column that has the filenames of images AND a label
    - In the case of multimodal data, the other columns (except these two) will be automatically processed
      - For example: filename will be cat1.png and label will be cat
    - You may NOT have paths in the filename
      - For example: /Users/smukherjee/Downloads/alfie_automl_engine/app/test.png is NOT ALLOWED.

  - A folder with images
    - The sub folders here will be your image files

- An example format is as belows
- Note that you MUST zip this file. ONLY .zip is accepted.

```
➜  imdb_subset git ✗ ls
images       metadata.csv
➜  imdb_subset git ✗ ls images
Action  Comedy  Horror  Romance
➜  imdb_subset git ✗ ls images/Action
tt10307440.jpg tt10308928.jpg tt12064810.jpg tt14873054.jpg tt3758814.jpg  tt3876910.jpg  tt7149730.jpg  tt8593824.jpg  tt9624766.jpg
➜  imdb_subset git ✗ cat metadata.csv| head -2
,movie_id,description,genre
1,tt1798632.jpg,"A young girl tries to understand how she mysteriously gained the power to set things on fire with her mind. After being experimented on by a secret government entity called The Shop, Andy McGee develops psychic powers and meets the love of his life. Together they have a daughter with a power of her own and The Shop will stop at nothing to get them back. In a flashback, baby Charlene ""Charlie"" McGee sits in her crib, spontaneously setting the room ablaze with her pyrokinesis power and sending her father Andrew ""Andy"" McGee into a panic. In another flashback, a young Andy and his girlfriend Victoria ""Vicky"" Tomlinson talk to a doctor in a clinical trial, who explains to them that they will be injected with the experimental chemical drug Lot, which secretly gives them supernatural powers Andy gains telepathy, and Vicky obtains telekinesis.In the present day, Charlie is sitting at the kitchen table after having a nightmare. Her parents join her and Charlie explains that she has been repressing something bad, her powers becoming more unstable. She unintentionally causes a ruckus at her school after exploding a bathroom stall due to anger at being bullied. Andy is shown using his power, ""the push"", to influence a client to stop smoking, although the strain causes his eyes to bleed.Meanwhile, in a secret facility, Captain Jane Hollister, leader of the Department of Scientific Intelligence DSI, is monitoring thermal signatures caused by Charlie's outbursts. She visits Doctor Joseph Wanless, creator of Lot and the resulting superhumans, who implores Hollister to terminate Charlie before her powers become uncontrollable. Hollister enlists fellow superhuman John Rainbird to assist her. Rainbird visits the McGee home, confronting Vicky, who attempts to counterattack with her repressed telekinetic powers. ",horror
➜  imdb_subset git:(feat/endpoints_for_dataset_descriptions) ✗
```
