import numpy as np
import pandas as pd
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("employee_leave_train.csv", index_col=0)
train, test, _, _ = train_test_split(train, train, test_size=0.18, random_state=41)

print(train.head())

f = ["Образование", "Год начала работы", "Город", "Уровень оплаты", "Возраст", "Пол", "Отстранения", "Опыт"]
cat_features = ["Образование", "Год начала работы", "Город", "Уровень оплаты", "Возраст", "Пол", "Отстранения", "Опыт"]

train_dataset = Pool(data=train[f], label=train["Увольнение"], cat_features=cat_features)
test_dataset = Pool(data=test[f], label=test["Увольнение"], cat_features=cat_features)


# model = CatBoostClassifier(iterations=10000, random_seed=3, eval_metric='F1', auto_class_weights="Balanced")
# model.fit(train_dataset,
#           eval_set=test_dataset,
#           verbose=False)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train[f], train["Увольнение"])

ttest = pd.read_csv("employee_leave_test.csv", index_col=0)
# pred = model.predict(ttest[f])
# ans = pd.DataFrame(pred)
# ans.to_csv("submission.csv", index=False, header=False)

# print(model.get_best_score())
