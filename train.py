#trains the model and saves, this is done offline and added to the repo
#hypers were found and validated in the notebook (diagnosis.ipynb in flaredown_data repo)

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.externals import joblib

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
    fileout = open('features.txt', 'w')
    fileout.write(features_string)

def reshapeSymptoms(df):
    # reshape and one-hot the symptoms
    symptoms = pd.get_dummies(df[(df['trackable_type'] == "Symptom") & (df['trackable_value'] != 0)],
                              columns=['trackable_name'])
    symptoms = symptoms.drop(['trackable_id', 'trackable_type', 'trackable_value'], axis=1)
    symptoms = symptoms.groupby(['user_id', 'checkin_date']).agg(numericOr).reset_index()

    return symptoms

def filterCondition(x):
    keep = True
    for value in x:
        if value in onceoffs:
            keep = False
    return keep


def createXY(df, symptoms):
    newdf = df[df['trackable_type'] == 'Condition'].groupby(['user_id', 'checkin_date'])['trackable_name'].agg(
        combineConditions).reset_index()
    newdf = newdf.merge(symptoms, on=['user_id', 'checkin_date'])
    newdf = newdf.drop(['user_id', 'checkin_date'], axis=1)
    newdf.to_csv('clean.csv') #write this out so we don't need to train again later
    X = newdf.drop('trackable_name', axis=1)
    Y = newdf['trackable_name'].apply(makeList)  # each row of Y is a list
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y)
    joblib.dump(mlb, 'mlb.pkl')
    return X,Y

def normAge(age, maxAge):
    return age / maxAge

def addAgeAndSex(symptoms, df):
    df.loc[df['age'].abs() > 90, 'age'] = 30
    maxAge = df['age'].max();
    print maxAge
    averageAge = normAge(df['age'].mean(), maxAge)
    symptoms['age'] = averageAge
    for userid in set(df['user_id']):
        symptoms.loc[symptoms['user_id'] == userid, 'age'] = normAge(df[df['user_id'] == userid]['age'].iloc[0], maxAge)
    symptoms['age'].fillna(averageAge, inplace=True)

    symptoms['isMale'] = 0
    symptoms['isFemale'] = 0
    for userid in set(df['user_id']):
        if df[df['user_id'] == userid]['sex'].iloc[0] == "male":
            symptoms.loc[symptoms['user_id'] == userid, 'isMale'] = 1
        if df[df['user_id'] == userid]['sex'].iloc[0] == "female":
            symptoms.loc[symptoms['user_id'] == userid, 'isFemale'] = 1

    return symptoms


def trainRidge():
    pcaobj = PCA(n_components=625) #got this number from notebook, see comment at top
    Y_train_transformed = pcaobj.fit_transform(Y_train)
    rig = Ridge(alpha=1)
    rig.fit(X_train, Y_train_transformed)

    return rig, pcaobj

df = pd.read_csv("export.csv")
df['checkin_date'] = pd.to_datetime(df['checkin_date'])

#grab list of supported symptoms, it helps to remove the symptoms only reported by one or two users
symptomslist = []
infile = open('symptom_list.txt', 'r')
for line in infile:
    symptomslist.append(line.split('   ')[0])

def filterer(x):
    xx = x[x['trackable_type'] == 'Symptom']
    for symptom in xx['trackable_name'].values:
        if symptom not in symptomslist:
            return False
    return True
#df = df.drop(['age', 'sex', 'country'], axis=1)
dfnew = df.groupby('user_id').filter(filterer)

condition_counts = df[df['trackable_type'] == 'Condition']['trackable_name'].value_counts()
onceoffs = list(condition_counts[condition_counts < 90].keys())
cleandf = df.groupby('user_id').filter(filterCondition)
print "deleting " + str(len(onceoffs)) + " users who have unique condition values"

symptoms = reshapeSymptoms(cleandf)
symptoms = addAgeAndSex(symptoms, cleandf)

print "splitting x and y"
X_train, Y_train= createXY(cleandf, symptoms)
X_train = X_train.drop('sex', axis=1)
X_train = X_train.drop('country', axis=1)
print X_train.head()
writeFeatureList(X_train)

print "training"
model, pcaobj = trainRidge()

print "writing models"
joblib.dump(model, 'model.pkl')
joblib.dump(pcaobj, 'pcaobj.pkl')