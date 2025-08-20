# CSAM

code for: "Boosting Adversarial Transferability through Adversarial Attack Enhancer" (Applied Science)

## Environment

- nvidia-tensorflow==1.15.4

- python==3.6
- nump==1.18
- scipy==1.2.1

## Experiment

### Dataset

Download [dataset](https://drive.google.com/open?id=1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) ，and put it into [dev_data/](https://github.com/jxdaily/CSA/tree/main/dev_data)

### Pre-trained model

download [Pre-trained models](https://drive.google.com/open?id=10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw)，and put them into [models/](https://github.com/jxdaily/CSA/tree/main/models)

### Generate adversarial examples

`python csam.py`

### Evaluation

`python eval.py`

## Acknowledgments

Reference：[SI-NI-FGSM](https://github.com/JHL-HUST/SI-NI-FGSM)




