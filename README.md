# NeurIPS 2024 - Lux AI Season3 4th Place Solution
This repository contains all scripts, including the training code for the 4th place solution in the kaggle NeurIPS 2024 - Lux AI Season 3 competition.  
For a detailed explanation of the solution, please refer to [this discussion](https://www.kaggle.com/competitions/lux-ai-season-3/discussion/569928).

## Setup
### Download episode data
Please refer to this [Notebook](https://www.kaggle.com/code/kuto0633/lux-ai-s3-download-episodes-from-meta-kaggle) to download the following Submission ID episode data.
- 42704976
- 42705163
- 43152191
- 43155694
- 43212163
- 43212846
- 43276830

Place the downloaded episode data as follows  
`datas/scraping/{submission_id}/json/{episode_id}.json`

### Build python environment
Install uv and run the following command.
```
$ uv sync
```

## Run
The following command executes the entire process, including data preparation, IL model training, and submission creation:
```
$ bash run_all.sh
```