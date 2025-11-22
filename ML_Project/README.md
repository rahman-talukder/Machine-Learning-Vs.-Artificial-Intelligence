Machine Learning Project Description:

In this project, a complete machine-learning pipeline was built to classify Titanic passengers as survived or not survived. The workflow began with loading the dataset from Google Drive into a Google Colab environment. The raw dataset contained categorical and numeric variables, along with missing values and irrelevant columns. Initial exploratory analysis included value counts for features such as Ticket, Cabin, Sex, Pclass, and Embarked, followed by visual inspection using Seaborn pair plots.

Pre-processing steps included dropping unnecessary columns (Name, Ticket, Cabin), encoding categorical features using LabelEncoder, and handling missing values—most notably filling missing Age values with the median and removing remaining rows with null entries. The cleaned dataset was then divided into feature variables (X) and target variable (Y), followed by a standard train-test split.

Multiple classification models were trained and evaluated: K-Nearest Neighbors (KNN), Decision Tree Classifier, and Support Vector Machine (SVM) with a linear kernel. Accuracy and confusion matrices were used to evaluate model performance. Among these models, the SVM performed the best with approximately 73% accuracy. Finally, the trained SVM model was saved using Python's `pickle` module for future prediction and deployment.
