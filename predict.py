import pandas as pd

def predict(decision, threshold1, threshold2):
    pred = ((decision - threshold1) > 0).astype(float)
    max_decision = decision.max(1)
    for k in range(pred.shape[0]):
        cut = max_decision[k] - threshold2
        idx = (decision[k, :] >= cut)
        pred[k, idx] = 1
    return pred

def predictFromModel(model, pcaobj, X_pred):
    prediction = model.predict(X_pred)
    pred = pcaobj.inverse_transform(prediction)
    return predict(pred,0.5, 0.1)

def symptomListToFeatures(symptomList, age, sex, features):

    for item in symptomList:
        if 'trackable_name_' + item['name'] in features:
            features['trackable_name_' + item['name']] = 1
    features['age'] = age / 87
    if sex.lower() == 'male':
        features['isMale'] = 1
    if sex.lower() == 'female':
        features['isFemale'] = 1
    return pd.DataFrame(features, index=[0])  #turns dict into a df with one row

def loadFeatures():
    filein = open('features.txt', 'r')
    featureDict = {}
    for feature in filein:
        if (feature.rstrip('\n') in featureDict):
            print feature
        featureDict[feature.rstrip('\n')] = 0
    return featureDict
