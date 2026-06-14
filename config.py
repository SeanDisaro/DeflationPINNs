import os
from pathlib import Path, PurePath


MY_DEVICE = "cpu"#"cuda:0"
MY_DEVICE_IDX = int(MY_DEVICE[-1]) if MY_DEVICE[-1]. isnumeric() else -1

repositoryPath: PurePath = os.getcwd()

pathTrainedModels: PurePath = PurePath(repositoryPath, "models")
pathData:PurePath = PurePath(repositoryPath, "data")
loggingFile: PurePath = PurePath(repositoryPath, "logging.log")
testsFolder: PurePath = PurePath(repositoryPath, "tests")
plotFolder: PurePath = PurePath(testsFolder, "pictures")
