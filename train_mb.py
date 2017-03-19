import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.externals import joblib
#from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier


#def makeList(x):
#    return x[5:-2].split(',')

def combineConditions(x):
    return set(x)

def makeList(x):
    return list(x)

def numericOr(x):
    if 1 in x.values:
        return 1
    else:
        return 0

def writeFeatureList(df):
    features_string = "\n".join(df.columns)
    fileout = open('features_rf.txt', 'w')
    fileout.write(features_string)

def reshapeSymptoms(df):
    # reshape and one-hot the symptoms
    symptoms = pd.get_dummies(df[(df['trackable_type'] == "Symptom") & (df['trackable_value'] != 0)],
                              columns=['trackable_name'])
    symptoms = symptoms.drop(['trackable_id', 'trackable_type', 'trackable_value'], axis=1)
    symptoms = symptoms.groupby(['user_id', 'checkin_date']).agg(numericOr).reset_index()
    return symptoms

def createXY(df, symptoms):
    df = df[df['trackable_type'] == 'Condition'].groupby(['user_id', 'checkin_date'])['trackable_name'].agg(
        combineConditions).reset_index()
    df = df.merge(symptoms, on=['user_id', 'checkin_date'])
    df = df.drop(['user_id', 'checkin_date'], axis=1)
    X = df.drop('trackable_name', axis=1)
    Y = df['trackable_name'].apply(makeList)  # each row of Y is a list
    #Y = df['trackable_name']
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y)
    joblib.dump(mlb, 'mlb_simple.pkl')
    return X,Y

def train(X, Y):
    clf = OneVsRestClassifier(MultinomialNB())
    clf = clf.fit(X, Y)

    return clf

def filterer(x):
    xx = x[x['trackable_type'] == 'Symptom']
    for symptom in xx['trackable_name'].values:
        if symptom not in symptomslist:
            return False
    return True

def filterCondition(x):
    keep = True
    for value in x:
        if value in onceoffs:
            keep = False
    return keep

#removing symptoms only listed by very few users
symptomslist = []
infile = open('symptom_list.txt', 'r')
for line in infile:
    symptomslist.append(line.split('   ')[0])

df = pd.read_csv("export.csv")
condition_counts = df[df['trackable_type'] == 'Condition']['trackable_name'].value_counts()
onceoffs = list(condition_counts[condition_counts < 90].keys())
cleandf = df.groupby('user_id').filter(filterCondition)
print "deleting " + str(len(onceoffs)) + " users who have unique condition values"


df = df.drop(['age', 'sex', 'country'], axis=1)
dfnew = df.groupby('user_id').filter(filterer)

X,Y = createXY(dfnew, reshapeSymptoms(dfnew))
writeFeatureList(X)
model = train(X,Y)
joblib.dump(model, 'model_mb.pkl')