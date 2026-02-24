### EfficientNet Training Usage
_Run using uv to automatically download dependencies and correct python version_

_By default pyproject.toml is setup to use CUDA with Nvdia GPUs, for CPU training remove lines 15-21 in the pyproject.toml file_

```uv run mergedtrainer.py [DATASET_INPUT DIRECTORY]```

_for additional parameters information run_

```uv run src/mergedtrainer.py -h```

### YOLO Training Usage
_Open leafdetection_training.py and edit paths and parameters as needed and run_

```uv run leafdetection_training.py```