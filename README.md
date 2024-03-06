![banner](https://images.aicrowd.com/raw_images/challenges/banner_file/1109/e848872d2871cbbf24b5.png)

> The repository is not actively maintained but issues can be raised to fix particular issues.

# Sound Demixing Challenge

This challenge is an opportunity for researchers and machine learning enthusiasts to test their skills on the difficult task of audio source separation: given an audio signal as input (referred to as ?mixture?), the challenge is to decompose it into its individual parts.

The main aim of source-separation is to separate vocals and specific intruments from a recorded song and facilitate audio professionals for remixing and remastering of the old songs.


## About the project

The project uses run-time augmentation method and some efficient modifications for different source separation backends. Please refer to the actual code for more info on the available backends. 

## Dataset
For the course of training, the project only uses [DnR dataset](https://zenodo.org/records/6949108)[1]. 


## Submission and results

The result on the public leaderboard is as follows:

| SDR Dialogue  | SDR Effect  | SDR Music  | 
|---|---|---|
| 8.948  | 1.224  | 1.442  |

This model achieved the best score for "speech" compared to any other entry for the CDX competitions. 

## Setting up the environment
The project has python version dependency of python3.9.18 as the base python version. The project also has two other strict dependencies:

1. torch==1.12.1
2. pytorch-lightning==1.8.1

To set up the environment, follow the steps:

1. Create Environment file `.env` in your project root directory with all the required variables. For base use, the values in the .env files could be as follows:

```bash
CUDA_VISIBLE_DEVICES="0"
HYDRA_FULL_ERROR=1
WANDB_API_KEY=""
```

2. Create Python environment as python 3.9.18 env
If you use Anaconda package manager or Miniconda, you can simply use the following command:

```bash
conda create -n sdx python=3.9.18 
```

3. Install requirements using the following command.
```bash
pip install -r requirements.txt
```
Please note that some installations might fails due to deprecation of old pytorch version. For that, please install pytorch_lightning and torch manually.

## Training
To train the model, you can run the `train.py`. This file uses hydra config management which is strictly dependent on the following file `config/train.yaml`.

```bash
python train.py
```

Please update train.yaml in config folder as per the requirement. I have tried to name all the variable as detailed as possible, but if something is still unclear, please raise an issue. 

One key change that will be required for switching between different challenge tracks is to efficiently modify the `train.yaml` with appropriate data. For changing the dataset, there is a key-value pair under `paths` variable in `dataset`. Please also change the evaluation dataset paths as well for replicate the competition evaluation locally. 

1. For CDX challenge change the respective value of the keys=["speech", "music", "sfx"] to their respective paths in glob format.

2. For MDX challenge, update the keys=["drums", "bass", "accompaniements", "vocals"] with values as their respective paths in the glob format.

The project lacks edge case tests and will fail if settings files are missing or the settings are incorrect, so please make sure all the files are present in the training dataset folder else the code will fail before strating training. 


## Inference

Inference codes are available at: https://gitlab.aicrowd.com/tushar_dhyani/sdx-2023-cinematic-sound-demixing-starter-kit

For the final submission, best models for each stem were used for the submission hash: `55838dc661d49480449bcf430e65803a881f8a08`. 

## References

```bibtex
[1] Petermann, D., Wichern, G., Wang, Z.-Q., & Le Roux, J. (2021). Divide and Remaster (DnR) (2.0)
```

## Cite

```
@misc{cdxsource2023,
  author = {Tushar Dhyani},
  title = {Sound Demixing challenge 2023: Runtime augmentations for efficient training of source separation models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Thanatoz-1/SoundDemixing2023}},
  commit = {4f57d6a0e4c030202a07a60bc1bb1ed1544bf679}
}
```