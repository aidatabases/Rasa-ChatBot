# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

from collections import defaultdict
import pandas as pd
import emoji
import pickle
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence, text
from keras.models import load_model



from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


# class ActionSearchTwitter(Action):

#     def name(self) -> Text:
#         return "action_twitter"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
#         entities=tracker.latest_message['entities']
#         print(entities)
        
#         for e in entities:
#             if e['entity']=='emotions':
#                 name=e['value']


#         dispatcher.utter_message(text=name)

#         return []
class ActionSearchTwitter(Action):
    
    def name(self) -> Text:
        return "action_emotion"

    # sent_to_id  = {"empty":0, "sadness":1,"excited":2,"fear":3,
    #                     "surprise":4,"disgust":5,"happiness":6,"anger":7}
    # contractions = pd.read_csv("contractions.csv")
    # cont_dic = dict(zip(contractions.Contraction, contractions.Meaning))

    # misspell_data = pd.read_csv("aspell.txt",sep=":",names=["correction","misspell"])
    # misspell_data.misspell = misspell_data.misspell.str.strip()
    # misspell_data.misspell = misspell_data.misspell.str.split(" ")
    # misspell_data = misspell_data.explode("misspell").reset_index(drop=True)
    # misspell_data.drop_duplicates("misspell",inplace=True)
    # miss_corr = dict(zip(misspell_data.misspell, misspell_data
    

    def clean_text(self,val):
        #val = self.misspelled_correction(val)
        val = self.cont_to_meaning(val)
        val = ' '.join(self.punctuation(emoji.demojize(val)).split())
        return val

    def misspelled_correction(self,val):
        misspell_data = pd.read_csv("aspell.txt",sep=":",names=["correction","misspell"])
        misspell_data.misspell = misspell_data.misspell.str.strip()
        misspell_data.misspell = misspell_data.misspell.str.split(" ")
        misspell_data = misspell_data.explode("misspell").reset_index(drop=True)
        misspell_data.drop_duplicates("misspell",inplace=True)
        miss_corr = dict(zip(misspell_data.misspell, misspell_data))
    
        for x in val.split(): 
            if x in miss_corr.keys(): 
                val = val.replace(x, miss_corr[x]) 
        return val

    def cont_to_meaning(self,val): 
        sent_to_id  = {"empty":0, "sadness":1,"excited":2,"fear":3,
                        "surprise":4,"disgust":5,"happiness":6,"anger":7}
        contractions = pd.read_csv("contractions.csv")
        cont_dic = dict(zip(contractions.Contraction, contractions.Meaning))
        for x in val.split(): 
            if x in cont_dic.keys(): 
                val = val.replace(x, cont_dic[x]) 
        return val

    def punctuation(self,val): 
        punctuations = '''()-[]{};:'"\,<>./@#$%^&_~'''
        for x in val.lower(): 
            if x in punctuations: 
                val = val.replace(x, " ") 
        return val
 
    with open('tokenizer.pickle', 'rb') as handle:
        token = pickle.load(handle)
    batch_size = 32
    model = load_model('my_model3.h5')
    max_len = 160

    

    def output2(self,text1):
        with open('tokenizer.pickle', 'rb') as handle:
            token = pickle.load(handle)
        batch_size = 32
        sent_to_id  = {"empty":0, "sadness":1,"excited":2,"fear":3,
                        "surprise":4,"disgust":5,"happiness":6,"anger":7}
        model = load_model('my_model3.h5')
        max_len = 160
        text = self.clean_text(text1)
        twt = token.texts_to_sequences([text1])
        twt = sequence.pad_sequences(twt, maxlen=max_len, dtype='int32')
        sentiment = model.predict(twt,batch_size=1,verbose = 2)

        result = pd.DataFrame([sent_to_id.keys(),sentiment[0]]).T
        result=result.dropna()
        result.columns = ["sentiment","percentage"]
        result=result.T
        result.columns = result.iloc[0]
        result.drop(['sentiment'],axis=0,inplace=True)
        result=result.iloc[:1]
        r = result.to_numpy().tolist()
        r=r[0]
        return r



    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        #print('LINK:',tracker.get_slot('emotions'))
        sent=tracker.latest_message['text']
        sent=sent.rsplit(' ', 1)[0]
        em=['empty','sadness','excited','fear','surprise','disgust','happiness','anger']
        print(type(sent))
        a=self.output2(sent)
        d={}
        for i,j in zip(em,a):
            d[i]=j
        print(d)


        
   

        dispatcher.utter_message(text=str(d))
        #dispatcher.utter_template("utter_info",tracker,link=a)

        return []
