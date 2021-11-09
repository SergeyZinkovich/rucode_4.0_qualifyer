from sklearn.model_selection import train_test_split
import pandas as pd
import os
import subprocess
import shutil
import pathlib

BASE_DIR = pathlib.Path(__file__).parent

df = pd.read_csv("train.csv")

train, test, _, _ = train_test_split(df, df, test_size=0.2, random_state=41)

for i in train.values:
    # os.system('copy data/train/' + i[0] + ' /data2/train/' + str(i[1]) + '/' + i[0])
    # subprocess.call('copy data/train/' + i[0] + ' /data2/train/' + str(i[1]) + '/' + i[0], shell=False)
    shutil.copyfile(os.path.join(BASE_DIR, 'data/train/' + i[0]), os.path.join(BASE_DIR, 'data2/train/' + str(i[1]) + '/' + i[0]))

for i in test.values:
    # os.system('copy data/train/' + i[0] + ' /data2/test/' + str(i[1]) + '/' + i[0])
    # subprocess.call('copy data/train/' + i[0] + ' /data2/test/' + str(i[1]) + '/' + i[0], shell=False)
    shutil.copyfile('data/train/' + i[0], 'data2/test/' + str(i[1]) + '/' + i[0])






