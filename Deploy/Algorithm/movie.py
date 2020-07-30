import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import tensorflow as tf
import sys
import pickle
import re
from tensorflow.python.ops import math_ops
from urllib.request import urlretrieve
from os.path import isfile, isdir
import zipfile
import hashlib

import collections 
import operator
from functools import reduce

import datetime
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2
import time

from flask import Flask,request
app = Flask(__name__)

def load_pkl(pkl_name):
    with open(pkl_name, 'rb') as f:
        return pickle.load(f)

#根据电影推荐同类型电影
#movieInfo页面调用，传入Id值，返回一个20长度的电影List
@app.route('/sametype')
def RecommendSameType():
    movie_id_val = int(request.args.get('id'))
    top_k = int(request.args.get('size'))
    return str(recommend_same_type_movie(movie_id_val,top_k))

#看过这个电影的人还看了（喜欢）哪些电影
#movieInfo页面调用，传入Id值，返回一个20长度的电影List 
@app.route('/other')
def RecommendOther():
    movie_id_val = int(request.args.get('id'))
    top_k = int(request.args.get('size'))
    return str(recommend_other_favorite_movie(movie_id_val,top_k))


def recommend_same_type_movie(movie_id_val,top_k=50):
    norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keepdims=True))
    normalized_movie_matrics = movie_matrics / norm_movie_matrics

    #推荐同类型的电影
    probs_embeddings = (movie_matrics[movieid_map[movie_id_val]]).reshape([1, 200])
    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
    sim = (probs_similarity.numpy())
    results = (-sim[0]).argsort()[0:top_k]

    result=[]
    
    for i in results:
        for k,v in movieid_map.items():
            if v==i:
                result.append(k)
    
    return result




def recommend_other_favorite_movie(movie_id_val,top_k=50):

    probs_movie_embeddings = (movie_matrics[movieid_map[movie_id_val]]).reshape([1, 200])
    probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
    favorite_user_id = np.argsort(probs_user_favorite_similarity.numpy())[0][-top_k:]

    probs_users_embeddings = (users_matrics[favorite_user_id-1]).reshape([-1, 200])
    probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
    sim = (probs_similarity.numpy())
    results = (-sim[0]).argsort()[0:top_k]
 
    #print("喜欢看这个电影的人还喜欢看：")
          
    result=[]
    
    for i in results:
        for k,v in movieid_map.items():
            if v==i:
                result.append(k)

    return result


#根据传回来的字符串拆分后推荐
#index页面调用，传入两个idlist，第一个是喜欢电影的list，第二个是浏览历史的list，返回一个20长度的电影List
@app.route('/both')
def recomment_both():
    string1 = request.args.get('likeids')
    string2 = request.args.get('historyids')
    historyIdList=string1.split(',')
    likeIdList=string2.split(',')
    result=[]
    for i in range(len(historyIdList)):
        result.append(recommend_same_type_movie(int(historyIdList[i]), 10))
        result.append(recommend_other_favorite_movie(int(historyIdList[i]), 10))
    for i in range(len(likeIdList)):
        result.append(recommend_same_type_movie(int(likeIdList[i]), 10))
        result.append(recommend_other_favorite_movie(int(likeIdList[i]), 10))

    result=reduce(operator.add, result)
    ctr = dict(collections.Counter(result))
    ctr=sorted(ctr.items(),key=lambda item:item[1],reverse=True)[0:100]
    results=[]
    for i in range(10):
        results.append(ctr[i][0])
    return str(results)

#index页面调用，传入喜欢电影的list，返回一个20长度的电影list
@app.route('/like')
def recomment_like():
    string = request.args.get('likeids')
    likeIdList=string.split(',')
    result=[]
    for i in range(len(likeIdList)):
        result.append(recommend_same_type_movie(int(likeIdList[i]), 10))
        result.append(recommend_other_favorite_movie(int(likeIdList[i]), 10))

    result=reduce(operator.add, result)
    ctr = dict(collections.Counter(result))
    ctr=sorted(ctr.items(),key=lambda item:item[1],reverse=True)[0:100]
    results=[]
    for i in range(10):
        results.append(ctr[i][0])
    return str(results)

#index页面调用，传入浏览历史的list，返回一个20长度的电影list
@app.route('/history')
def recomment_history():
    string = request.args.get('historyids')
    historyIdList=string.split(',')
    result=[]
    for i in range(len(historyIdList)):
        result.append(recommend_same_type_movie(int(historyIdList[i]), 10))
        result.append(recommend_other_favorite_movie(int(historyIdList[i]), 10))

    result=reduce(operator.add, result)
    ctr = dict(collections.Counter(result))
    ctr=sorted(ctr.items(),key=lambda item:item[1],reverse=True)[0:100]
    results=[]
    for i in range(10):
        results.append(ctr[i][0])
    return str(results)

if __name__ == "__main__":
    movieid_map=load_pkl('data/movieid_map.pkl')
    userid_map=load_pkl('data/userid_map.pkl')
    genres2int=load_pkl('data/genres2int.pkl')


    movies_unchange=pd.read_csv('data/movies_unchange.csv',encoding='utf-8')
    users_unchange=pd.read_csv('data/users_unchange.csv',encoding='utf-8')
    users=pd.read_csv('data/users_unchange.csv',encoding='utf-8')
    movies=pd.read_csv('data/movies_int.csv',encoding='utf-8')

    network = tf.keras.models.load_model('data/model.h5')
    movie_matrics = pickle.load(open('data/movie_matrics.p', mode='rb'))
    users_matrics = pickle.load(open('data/users_matrics.p', mode='rb'))

    users[['USER_MD5']] = users[['USER_MD5']].astype(int)
    users_unchange[['USER_MD5']] = users_unchange[['USER_MD5']].astype(int)

    movies_orig=pd.DataFrame(movies_unchange.values)
    users_orig=pd.DataFrame(users_unchange.values) 
    
    app.run(host="0.0.0.0",port="5000")

    #if sys.argv[1]=='movie':
    #    result=recommend_same_type_movie(int(sys.argv[2]), int(sys.argv[3]))
    #elif sys.argv[1]=='user':
    #    result=recommend_your_favorite_movie(int(sys.argv[2]), int(sys.argv[3]))
    #elif sys.argv[1]=='likeUser':
    #    result=recommend_other_favorite_movie(int(sys.argv[2]), int(sys.argv[3]))

    #print(result)

    


