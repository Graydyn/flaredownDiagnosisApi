from flask import Flask, request, Response
from flask_restplus import Resource, Api, fields
import sys
import predict
import json
from flask_cors import CORS
from sklearn.externals import joblib

app = Flask(__name__)
CORS(app)

api = Api(app, version='1.0', title='Flaredown Diagnosis API', description='Attempts to diagnosis chronic illness')

symptom = api.model('Symptom', {
    'name': fields.String(required=True, example='Fatigue', description='Symptom name'),
})


allData = api.model('data', {
    'symptoms': fields.List(fields.Nested(symptom)),
    'sex': fields.String(required=False, example='female', description='male, female, other'),
    'age': fields.Integer(readOnly=True, example=32, description='The users age')
})


model = joblib.load('model.pkl')
pcaobj = joblib.load('pcaobj.pkl')
mlb = joblib.load('mlb.pkl')

def loadFeatures():
    filein = open('features.txt', 'r')
    featureDict = {}
    for feature in filein:
        if (feature.rstrip('\n') in featureDict):
            print feature
        featureDict[feature.rstrip('\n')] = 0
    return featureDict

featureDict = loadFeatures()


@api.errorhandler
def default_error_handler(e):
    message = 'An unhandled exception occurred.'

    return {'message': message}, 500

@api.route('/main')
@api.response(200, 'Success.')
class Generic(Resource):
    @api.doc(description='')
    @api.expect(allData)
    def post(self):

      print request.json
      symptoms = request.json['symptoms']
      age = request.json['age']
      sex = request.json['sex']

      userDf = predict.symptomListToFeatures(symptoms, age, sex, featureDict)

      predictions, confidences = predict.predictFromModel(model, pcaobj, userDf, mlb)
      predsList = []
      for i in range(0,len(predictions[0])):
          prediction = predictions[0][i]
          confidence = confidences[i]
          predsList.append({
              "name" : prediction,
              "confidence" : confidence*100
          })

      responseJson = json.dumps(predsList)

      return responseJson, 200, {'Access-Control-Allow-Origin' : '*'}



if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0',port=5000)
    #app.run(debug=True)
