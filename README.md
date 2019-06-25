# Falls prevention with RNNs

This repository contains scripts responsible for training, tuning and testing recurrent neural networks capable of detecting presynkope state of patients based on their medical temporal data (heart rate, mean blood pressure).

# Requirements
```
python >= 3.6.2
numpy >= 1.15.0
pandas >= 0.23.4
scipy >= 1.1.0
pytorch >= 0.4.1
```

# Training
Model may be trained on training data set. This script requires 2 arguments:

*	*hiddenSize* - number of neurons in hidden layers,

*	*depth* - number of stacked RNNs on top of each other.

Every epoch snapshots are saved to **checkpoints/** and models are evaluated, creating coresponding files in **csv/** directory. This script periodically prints loss.
```
python trainer.py hiddenSize depth
``` 
Training process may be resumed from a checkpoint.
```
python retrainer.py modelName
```

# Inspecting models
Model inspector prints basic information concerning a particular model. Model's name is the filename in **checkpoints/** without *.model* extension.
```
python model_inspector.py modelName
```

# Testing
Tester evaluates models and creates coresponding files in **csv/**.
```
python tester.py modelName
```

# Tuning
Best hyperparameters can be found by running the tuner. After trying 50 points it saves its state in **checkpoints/** with extension *.tuner*.
```
python tuner.py
```
In case of having at least 1 tuner snapshot the process can be resumed.
```
python tuner.py tunerName1 tunerName2 ...
```
In order to inspect an existing tuner snapshot:
```
python tuner_inspector.py tunerName
```

# Data visualization
In order to plot availible data:
```
python preview.py
```
In order to view contents of a csv file (smoothingFactor defines how many samples are averaged):
```
python visualize.py pathToCsv smoothingFactor
```

**Whenever smoothingFactor equal to 0 is provided the script plots just the maximum values achieved by the series.**

# Data sets generation
Script below balances and divides data into 3 sets (70% training, 15% validation and test).
```
python divide.py 70 15 15
```

# Performance evaluation
First call searches the best possible threshold, the second one only tries the given threshold.
```
python conf_matrix.py pathToCsv
python conf_matrix.py pathToCsv threshold
```

# Misc
dataset_inspector.py - prints dataset information

other_methods.py - calculates scores for the classical classification algorithms

plot_loss.py - prints loss (in case it was forwared to a file from trainer.py)

mat_to_csv.py - if "Mat/" directory exists then csv data may be generated from matlab files

