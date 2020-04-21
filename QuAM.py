import os
import pandas as pd
import numpy as np
from joblib import load

def main():
    print("Welcome to our DNA analyzing QuAM")
    print("Please write the name of the file with your dataset: ")
    f = input()
    c_type = input("Classify the whole dataset? [y/n]? ")
    if c_type == "y":
        output = "result.txt"
        try:
            os.remove(output)
        except OSError:
            pass
        
        data = pd.read_csv(f)
        dna = data.iloc[:, :-1]
        
        classifier = load('./models/finalData_SVM_classifier.joblib')
        p_class = classifier.predict(dna)
        
        with open(output, "a") as my_file:
            for i, c in enumerate(p_class):
                my_file.write("record {} predicted class: {}\n".format(i, c))
                
        print("Done. Results outputted in {}".format(output))
    else:
        print("Enter a row number for which DNA you would like to classify (0-indexed): ")
        n = int(input())
    
        data = pd.read_csv(f)
        dna = data.iloc[n, :-1]
        
        classifier = load('./models/finalData_SVM_classifier.joblib')
        p_class = classifier.predict(np.array(dna).reshape(1, -1))
        
        print("Predicted class: {}".format(p_class))

if __name__ == "__main__":
    main()