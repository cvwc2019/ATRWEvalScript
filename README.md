# ATRWEvalScript
 GroundTruth & Eval Scripts for ATRW Dataset

```
ATRWEvalScript
 ├─annotations
 |   (ground-truth files)
 ├─atrwtool
 |   (evaluation scripts)
 ├─sampleinput
 |   (sample input files for testing scripts & refernce)
 ├─README.md
```

## Install

The scripts run under python3. 

Shell command bellow installs required libs.

`pip3 install -r ./atrwtool/requirments.txt`

## Usage

`python3 ./atrwtool/main.py [task] [input file path]`

Where `[task]` is one of `detect, pose, plain, wild` for corresponding track.

For detailed result file format description, please refer to **Format Description** at https://cvwc2019.github.io/challenge.html



