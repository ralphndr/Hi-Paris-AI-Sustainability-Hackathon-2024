import pandas as pd 
from sklearn.preprocessing import LabelEncoder

X = pd.read_csv("D:/HiParis_Hackathon/hickathon/sample_data/Num_data_cleaned.csv")

y = pd.read_csv("D:/HiParis_Hackathon/hickathon/sample_data/y_train.csv")


# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the y_train column
y = label_encoder.fit_transform(y.iloc[:, 0])  # Replace 0 with the specific column index if needed

# Print the encoded labels
print("Encoded Labels:")
print(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
cf = RandomForestClassifier(n_estimators= 25, criterion='entropy')
cf.fit(x_train,y_train)

y_pred = cf.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print("Train Score:", cf.score(x_train,y_train))
print("Test Score:", cf.score(x_test,y_test))