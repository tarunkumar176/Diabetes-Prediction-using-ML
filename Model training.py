import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import joblib

#loading the dataset
df=pd.read_csv("dataset.csv")

features=['HighChol','BMI','HighBP','HeartDiseaseorAttack',
          'PhysActivity', 'Fruits', 'PhysHlth', 'DiffWalk',
          'HvyAlcoholConsump','GenHlth','Sex','Age'] #collecting the best attributes for the model

x=df[features]
y=df['Diabetes_binary']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)
y_flase,y_true=(y_train==0).sum(),(y_train==1).sum()
scale=y_flase/y_true

# using random forest
rondom_forest_model=RandomForestClassifier(class_weight='balanced', random_state=42)
rondom_forest_model.fit(x_train,y_train)
rf_pred=rondom_forest_model.predict(x_test)

print("accuracy= ",accuracy_score(y_test, rf_pred))
print("confusion matrix= ",confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

#decision tree
dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train,y_train)
dt_pred=dt_model.predict(x_test)

print("accuracy",accuracy_score(y_test,dt_pred))
print("confusion matrix= ",confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# using XGBoost
xg_model=XGBClassifier( max_depth=4,
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    use_label_encoder=False,
    eval_metric='logloss')
xg_model.fit(x_train,y_train)
xg_pred=xg_model.predict(x_test)

print("accuracy",accuracy_score(y_test,xg_pred))
print("confusion matrix= ",confusion_matrix(y_test, xg_pred))
print(classification_report(y_test, xg_pred))


joblib.dump(rf_pred, 'model.pkl')
joblib.dump(dt_pred, 'model.pkl')
joblib.dump(xg_model, 'model.pkl')