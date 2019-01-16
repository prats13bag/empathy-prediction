import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    import imp
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.dummy import DummyClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn import model_selection
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report


class train:
    def loadData():
        print ("Loading data...")
        responses=pd.read_csv('responses.csv')
        print ("Data Loaded. Shape of the loaded data is", responses.shape)
        return responses

    def analyseData(responses):
        print ("###########################################################################")
        print ("############### Following is found regarding missing values ###############")
        print ("###########################################################################")
        print ("Non-Numeric Columns\t  Count of Null Values")
        print (responses.select_dtypes(include=['object']).isnull().sum(axis=0).to_string())
        print ("###########################################################################")
        print ("Number of NULL Values in Target Class (Empathy Column) =", responses.shape[0] - responses.groupby('Empathy').size().sum(axis=0))
        missing_value_count_list = responses.select_dtypes(exclude=['object']).isnull().sum(axis=0).values
        count=0
        for i in missing_value_count_list:
            if i!=0:
                count+=1
        print ("Number of Numeric Columns having atleast one missisng value =", count)
        print ("###########################################################################")
        
    def imputeAndEncode(responses):
        print ("Imputating missing values...")
        responses = responses.fillna(responses.mode().iloc[0])
        print ("Missing values replaced by most frequent value in that column")
        print ("Performing one-hot encoding...")
        for col in responses.columns:
            if responses[col].dtype.kind=='O':
                responses[col]=responses[col].astype('category')
        df = pd.get_dummies(responses, columns=list(responses.select_dtypes(include=['category']).columns))
        print ("Shape of the dataset after one-hot encoding is",df.shape)
        print ("###########################################################################")
        return df
       
    def splitDataset(df):
        Y = df['Empathy']
        X = df.drop('Empathy', 1)
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.35)
        print ("Dataset is splitted into 65% training data and 35% testting data")
        print ("###########################################################################")
        return Xtr, Xte, Ytr, Yte
        
    def findOptimalClassifier(Xtr, Xte, Ytr, Yte):
        print('Trying different classifiers with default hyperparameters:')
        classifiers = {
            "Baseline Classifier": DummyClassifier(strategy='most_frequent', random_state=1),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=1),
            "Logistic Regression": LogisticRegression(solver='liblinear', multi_class='ovr', random_state=1),
            "SVM - linear kernel": SVC(kernel='linear', random_state=1),
            "SVM - degree 3 polynomial kernel": SVC(kernel='poly', gamma='auto', random_state=1),
            "SVM - rbf kernel": SVC(kernel='rbf', gamma='auto', random_state=1),
            "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=1),
            "AdaBoost Classifier": AdaBoostClassifier(random_state=1),
            "Multi Layer Perceptron Classifier": MLPClassifier(solver='lbfgs', alpha=0.9, max_iter=250, random_state=1),
        }

        maxScore=0
        desiredClassifier = str()
        for i in classifiers :
            clf = classifiers[i]
            clf = clf.fit(Xtr, Ytr)
            pred = clf.predict(Xte)
            score = round(accuracy_score(Yte, pred), 3)
            print ("=> Accuracy score on test data for", i,"is",score)
            if score>maxScore:
                maxScore = score
                desiredClassifier = i
        print ("\n###########################################################################")
        print ("The best performing classifier on this dataset is", desiredClassifier)
        print ("Accuracy corresponding to this classifier =", maxScore)
        print ("###########################################################################")
        return desiredClassifier
    
    def tuneGBMhyperparameters(folds=5):
        print("Tuning hyperparameters using GridSearchCV. This may take long time to complete")
        print("############################################################################")
        selected_model = GradientBoostingClassifier(random_state=1)
        params = {'max_depth':range(3,8,2),
                  'min_samples_split':range(200,401,50),
                  'max_features':range(25,31,5),
                  'subsample':[0.7,0.75,0.8]}

        # Running 4 jobs in parallel
        # using 5-fold cross-validation technique
        grid_search_model = GridSearchCV(selected_model,param_grid=params,n_jobs=4,iid=True,cv=folds)
        return grid_search_model
    
    def saveModel(final_clf, Xte, Yte):
        pickle_obj=[final_clf, Xte, Yte]
        empathy_prediction_model=open('empathy_prediction_model.pkl','wb')
        pickle.dump(pickle_obj,empathy_prediction_model)
        empathy_prediction_model.close()
        print("Model is saved as empathy_prediction_model.pkl")
        print("Training process is completed. You can now test the model. Buhbyee!")
        exit()
        
responses = train.loadData()
train.analyseData(responses)
df = train.imputeAndEncode(responses)
Xtr, Xte, Ytr, Yte = train.splitDataset(df)
opt_clf = train.findOptimalClassifier(Xtr, Xte, Ytr, Yte)
folds=5
if opt_clf == "Gradient Boosting Classifier":
    grid_search_model = train.tuneGBMhyperparameters(folds)
    grid_search_model = grid_search_model.fit(Xtr, Ytr)
    print("###########################################################################")
    print("Optimal hyperparameters found")
    print("###########################################################################")
    print("\nTraining is performed using the following optimal set of hyperparameters")
    print(grid_search_model.best_params_)
    print("###########################################################################")
    train.saveModel(grid_search_model, Xte, Yte)
else:
    print ("Exiting because of issues in finding GBM as base optimal classifier!\nPlease run again")
    exit()