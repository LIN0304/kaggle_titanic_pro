# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
# 堆疊模型
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# 加載數據集
train_data_path = '/kaggle/input/titanic/train.csv'
test_data_path = '/kaggle/input/titanic/test.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# 數據預處理和特徵工程
# 填充缺失值
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data['Has_Cabin'] = ~train_data['Cabin'].isnull()

# 提取稱謂
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
train_data['Title'] = train_data['Title'].map(title_mapping)
train_data['Title'] = train_data['Title'].fillna(0)

# 創建家庭大小特徵
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

# 刪除不需要的列
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 編碼類別特徵
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)

# 進一步的特徵工程
# 創建稱謂分組特徵
train_data['IsAlone'] = 0
train_data.loc[train_data['FamilySize'] == 1, 'IsAlone'] = 1

# 分箱 'Age' 和 'Fare'
train_data['AgeBin'] = pd.cut(train_data['Age'], 5)
train_data['FareBin'] = pd.qcut(train_data['Fare'], 4)

# 轉換分箱為數字標籤
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
train_data['AgeBin_Code'] = label.fit_transform(train_data['AgeBin'])
train_data['FareBin_Code'] = label.fit_transform(train_data['FareBin'])

# 更新特徵集合
drop_elements = ['PassengerId', 'Age', 'Fare', 'AgeBin', 'FareBin']

# 檢查這些列是否在 DataFrame 中，如果在則刪除
train_data = train_data.drop(columns=[col for col in drop_elements if col in train_data.columns])

# 更新訓練和測試數據集
X_train = train_data.drop(['Survived'], axis=1)
y_train = train_data['Survived']

# 分割數據
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 超參數調優
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 定義基學習器
base_learners = [
                 ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                 ('svr', SVC(probability=True, kernel='linear', C=0.025)),
                 ('dt', DecisionTreeClassifier())
                ]

# 定義最終學習器
stacked_model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())

# 訓練堆疊模型
stacked_model.fit(X_train, y_train)

# 使用最佳模型進行預測
#best_grid = grid_search.best_estimator_

# 預處理測試數據集
test_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
test_data['Has_Cabin'] = ~test_data['Cabin'].isnull()
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Title'].map(title_mapping)
test_data['Title'] = test_data['Title'].fillna(0)
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)
X_test_final = test_data[X_train.columns]

# 預測
predictions = best_grid.predict(X_test_final)

# 創建提交文件
submission = pd.DataFrame({
    "PassengerId": test_data['PassengerId'],
    "Survived": predictions
})

# 輸出結果到 CSV 文件
submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)

print(f'Submission file saved to {submission_path}')
