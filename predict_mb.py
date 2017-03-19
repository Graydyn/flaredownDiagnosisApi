import pandas as pd
import math

#normalizing in attempt to turn these values into something that would mean something to a user
def normalizeCertainty(x, max):
    return x / (float(math.ceil(max * 100)) / 100.0)


def predictFromModel(model, X_pred, mlb):
    conf = model.predict_proba(X_pred)
    labels = []
    confidences = []
    for i in range(len(conf[0])):
        confidence = conf[0][i]
        if confidence != 0:
            labels.append(mlb.classes_[i])
            confidences.append(confidence)
    return labels,confidences

def symptomListToFeatures(symptomList,features):

    for item in symptomList:
        if 'trackable_name_' + item['name'] in features:
            features['trackable_name_' + item['name']] = 1
    return pd.DataFrame(features, index=[0])  #turns dict into a df with one row

def loadFeatures():
    filein = open('features_rf.txt', 'r')
    featureDict = {}
    for feature in filein:
        if (feature.rstrip('\n') in featureDict):
            print feature
        featureDict[feature.rstrip('\n')] = 0
    return featureDict
