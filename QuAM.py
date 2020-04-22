import os
import pandas as pd
import numpy as np
from joblib import load

names = ["Cryons", "Baltan", "Allasomorph", "Deltans", "Elves", "Florauna", "Geonosian", "Heptapods", "Iconians", "Jaridians", "Kaleds", "Lepidopterran", "Martians", "Na'vi", "Organians", "Petrosapien", "Quintesson", "Risians", "Sheliak", "Tatanga", "Unas", "Vaxasaurian"]

def main():
    print("Welcome to our DNA analyzing QuAM")
    print("Please write the name of the file with your dataset: ")
    f = input()
    c_type = input("Classify the whole dataset? [y/n]? ")
    if c_type == "y":
        output = "result.csv"
        try:
            os.remove(output)
        except OSError:
            pass
        
        data = pd.read_csv(f)
        target = data.iloc[:, -1]
        dna = data.iloc[:, :-1]
        
        classifier = load('./models/finalData_SVM_classifier.joblib')
        p_class = classifier.predict(dna)
        
        with open(output,'w') as file:
            file.write("record, actual, predicted, classification\n")
            for i, c in enumerate(p_class):
                file.write("{}, {}, {}, {}\n".format(i, names[target[i]], names[c], target[i] == c))
                
        print("Done. Results outputted in {}".format(output))
    else:
        c = input("Enter a class of an alien you would like to classify (type [s] to see list): ")
        while (c == "s"):
            print(names)
            c = input("Enter a class of an alien you would like to classify (type [s] to see list): ")
       
        data = pd.read_csv(f)
        dna = data.loc[data['target'] == names.index(c)]
        dna = dna.sample(n=1)
        print("Picked row {}:\n{}".format(dna.index[0], dna))
        dna = dna.iloc[:, :-1]
        
        classifier = load('./models/finalData_SVM_classifier.joblib')
        p_class = classifier.predict(np.array(dna).reshape(1, -1))
        
        print("Predicted class: {}".format(names[p_class[0]]))

if __name__ == "__main__":
    main()