import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer

import pickle


def load_data(database_filepath):
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('data',con=engine)
   
    X = df.iloc[:,0].values
    Y = df.iloc[:,3:]
    category_names = Y.columns
    Y = Y.values
    
    return X, Y, category_names


def tokenize(text):
    
    text = re.sub(r"[^a-zA-Z0-9]", " ",text).lower()
    text = word_tokenize(text)
    text =  [w for w in text if w not in stopwords.words("english")]
    text = [PorterStemmer().stem(w) for w in text]
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    
    return text


def preprocess(X):
    
    vectorizer = CountVectorizer(tokenizer=tokenize)
    X = vectorizer.fit_transform(X)
    tfidf = TfidfTransformer()
    X = tfidf.fit_transform(X)
    
    return X, vectorizer, tfidf

def preprocess_test(X, vectorizer, tfidf):

    X = vectorizer.transform(X)
    X = tfidf.transform(X)
    
    return X

def train(X, Y):
    clfs = [] 
    error_cols = []
    
    num_col = Y.shape[1]
   
    for i in range(num_col):
 
        print('i:', i)
        clf = LogisticRegression(class_weight='balanced')
        #clf = RandomForestClassifier()
        
        try:
            clf.fit(X, Y[:,i])
        except Exception as e:
            print('col:', i, ' caused error:', e)
            error_cols.append(i)
        clfs.append(clf)
    return clfs, num_col, error_cols

def pipeline_orig(X, y): 
    
    X, vectorizer, tfidf = preprocess(X)
    clfs, num_col, error_cols = train(X, y)
    return clfs, num_col, error_cols, vectorizer, tfidf


def predict(clfs, X, num_col, vectorizer, tfidf, error_cols=None):
    
    result = []
    
    X_prep = preprocess_test(X, vectorizer, tfidf)
    
    for i in range(num_col):      
        print('i:',i)
       
        if i in error_cols:        
            pred = np.zeros(len(X))
        else:
            clf = clfs[i]  
            pred = clf.predict(X_prep)
        result.append(pred)
     
    result = np.array(result)
    result = np.transpose(result)
    
    return result

def build_model(model_path):
     
    model = pickle.load(open(model_path, 'rb'))
    return model


def evaluate_model(y_pred, y_test, category_names):
    
   
    try: 
        num_col = num_col = y_pred.shape[1]
    except: 
        num_col = 1
        
    labels = category_names
    
    accuracies = [] 
    fvalues = []
    
    for i in range(num_col):
        
        try:
            pred = y_pred[:,i]
            test = Y_test[:,i]
        except:
            pred = y_pred
            test = y_test
        
        tp = np.logical_and(test, pred).sum()
        fp = np.logical_and(1-test, pred).sum()
        tn = np.logical_and(1-test, 1-pred).sum()
        fn = np.logical_and(test, 1-pred).sum()

        acc = (tp + tn) / (tp + fp + tn + fn)
    
        recall = tp / (tp + fn)
        prec = tp / (tp + fp)
        fval = 2 * recall * prec / (recall + prec)
                
        conf_matrix = [[tp, fp],[fn, tn]]
    
        print('label:', labels[i] ,'\naccuracy:', acc, '\nf-value:', fval)
        print('confusion matrix:')
        print(conf_matrix[0])
        print(conf_matrix[1])
        
        print('\n\n')
        
        accuracies.append(acc)
        fvalues.append(fval)
        
    
    print('average accuracy:', sum(accuracies) / len(accuracies))
    
    fvalues = [x for x in fvalues if x == x]
    
    ave_fvalues = sum(fvalues) / len(fvalues)
    
    print('average f score:', ave_fvalues)
    #print("Labels:", labels)
    #print("Accuracy:", accuracy)
    #print("Confusion Matrix:\n", confusion_mat)
    
    return ave_fvalues


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        #model = build_model(model_filepath)
        
        print('Training model...')
        clfs, num_col, error_cols, vectorizer, tfidf = pipeline_orig(X_train, Y_train)

        print('category names:', category_names)

        print('Evaluating model...')
        Y_test = Y_test
       
        Y_pred = predict(clfs, X_test, num_col, vectorizer, tfidf, error_cols)
        evaluate_model(Y_pred, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(clfs, model_filepath)
        pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
        pickle.dump(tfidf, open('models/tfidf.pkl', 'wb'))
        

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()