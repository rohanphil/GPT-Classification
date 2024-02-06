import pandas as pd 
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from dotenv import load_dotenv, find_dotenv
import json
import numpy as np

class predict_sentiment(object):

    """
    Class to predict and score data for sentiment analysis.

    """

    def __init__(self, data, text_column, target_column, classes):
        # add assertion that the data be a pandas dataframe\

       _ = load_dotenv(find_dotenv())

       openai.api_key=os.getenv("CHAT_GPT_API_KEY")
       self.text_column = text_column
       self.target_column = target_column
       self.classes = classes
       self.data = data

    def predict_one(self,x):
        text = x[self.text_column]
        prompt = f"\nClassify the following movie reviews into one of these classes {self.classes}"
        context = text + prompt
        function = function = {
                    "name": "predict_sentiment",
                    "description": "Predict the sentiment of a given review",
                    "parameters": {
                        "type" : "object",
                        "properties" : {
                        "prediction": {
                            "type": "string",
                            "enum": ["0", "1"]
                        }
                    },
                    },
                    "required": ["prediction"]
                }
        r = openai.chat.completions.create(
                model="gpt-4",
                temperature=0.0,
                messages=[{"role": "user", "content": context}],
                functions=[function],
                function_call={"name": "predict_sentiment"}
                )
        return json.loads(r.choices[0].message.function_call.arguments)['prediction']
    
    def predict(self):
        self.data['prediction'] = self.data.apply(lambda x : self.predict_one(x), axis = 1)
    
        return self.data
    
    def score(self):
        
        data = self.predict()
        data['prediction'] = data['prediction'].astype(int)
        return np.mean(data['prediction'] == data[self.target_column]) *100

        


