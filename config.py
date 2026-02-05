import os
from pathlib import Path, PurePath


repositoryPath: PurePath = os.getcwd()

pathTrainedModels: PurePath = PurePath(repositoryPath, "models")
pathData:PurePath = PurePath(repositoryPath, "data")
loggingFile: PurePath = PurePath(repositoryPath, "logging.log")
testsFolder: PurePath = PurePath(repositoryPath, "tests")
plotFolder: PurePath = PurePath(testsFolder, "pictures")
