import imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class test:
    
    def unpickleObjects():
        print ("Loading classifier...")
        empathy_prediction_file=open('empathy_prediction_model.pkl','rb')
        obj=pickle.load(empathy_prediction_file)
        print ("Details of the clasifier loaded can be found below:\n")
        print (obj[0])
        return obj[0],obj[1],obj[2]
    
    def predict(trained_clf, Xte, Yte):
        labels = ['1','2','3','4','5']
        print ("\nNow Predicting...")
        prediction = trained_clf.predict(Xte)
        print ("Prediction Completed. Find results below:")
        final_score = accuracy_score(Yte, prediction)
        print ("#################################################################")
        print ("##################### Classification Report #####################")
        print ("#################################################################")
        print (classification_report(Yte, prediction,target_names=['Not Very Empathetic (1)',
                                                           'Not Very Empathetic (2)',
                                                           'Not Very Empathetic (3)',
                                                           'Very Empathetic (4)',
                                                           'Very Empathetic (5)']))
        print ("#################################################################")
        print ("####################### Confusion  Matrix #######################")
        print ("#################################################################")
        cm = confusion_matrix(Yte, prediction)
        test.printConfusionMatrix(cm,labels)
        print ("#################################################################")
        print ("Prediction accuracy on test data is",final_score)
        print ("#################################################################")
        print ("Press ctrl+c to terminate")
        test.plotConfusionMatrix(cm)

    def plotConfusionMatrix(cm):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap=plt.cm.Reds)
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        ax.set_xticklabels(['1', '2', '3', '4', '5'])
        ax.set_yticklabels(['1', '2', '3', '4', '5'])
        for i in range(5):
            for j in range(5):
                text = ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        fig.tight_layout()
        plt.show()
        
    def printConfusionMatrix(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
        columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
        empty_cell = " " * columnwidth
        fst_empty_cell = (columnwidth-3)//2 * " " + "T/P"
        if len(fst_empty_cell) < len(empty_cell):
            fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
        print("    " + fst_empty_cell, end=" ")
        for label in labels:
            print("%{0}s".format(columnwidth) % label, end=" ")
        print()
        for i, l in enumerate(labels):
            print("    %{0}s".format(columnwidth) % l, end=" ")
            for j in range(len(labels)):
                cell = "%{0}.1f".format(columnwidth) % cm[i, j]
                if hide_zeroes:
                    cell = cell if float(cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                print(cell, end=" ")
            print()

clf, Xte, Yte = test.unpickleObjects()
test.predict(clf, Xte, Yte)