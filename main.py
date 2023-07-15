import tqdm
import json
import os
import scipy
import nltk
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from time import sleep
from nltk.corpus import stopwords

global it
it = 0

def make_tf_idf(u, s):
    global it
    print(f"Making tf-idf matrix {it + 1}/{BATCH_NUM}...")
    tf_idf = u @ np.diag(s)
    it += 1
    return tf_idf

TEST = False
version = 5
print("Enter the version of the model:")
print("1. small     (1 GB RAM required)")
print("2. medium    (5 GB RAM required)")
print("3. large     (10 GB RAM required)")
print("4. complete  (18 GB RAM required)")
if False:
    print("5. test")
version = int(input("Version: "))

if version == 1:
    BATCH_NUM = 1
elif version == 2 or version == 5:
    BATCH_NUM = 5
elif version == 3:
    BATCH_NUM = 10
elif version == 4:
    BATCH_NUM = 18

print("Loading data...")
TITLES = [np.load(f"./svd_matrix/titles/{'test_' if TEST else ''}titles_{i}.pkl", allow_pickle=True) for i in range(BATCH_NUM)]
VOC = [np.load(f"./svd_matrix/voc/{'test_' if TEST else ''}voc_{i}.pkl", allow_pickle=True) for i in range(BATCH_NUM)]
U = [np.load(f"./svd_matrix/u/{'test_' if TEST else ''}u_{i}.npy") for i in range(BATCH_NUM)]
S = [np.load(f"./svd_matrix/s/{'test_' if TEST else ''}s_{i}.npy") for i in range(BATCH_NUM)]
V = [np.load(f"./svd_matrix/v/{'test_' if TEST else ''}v_{i}.npy") for i in range(BATCH_NUM)]

US = [make_tf_idf(u, s) for u, s in zip(U, S)]
print("Loading complete!\n")

while 1:
    prompt = input("Enter a query: ")
    if prompt != ":q":
        prompt = prompt.lower()
        prompt = ''.join([c for c in prompt if c.isascii()])
        prompt = nltk.word_tokenize(prompt)
        prompt = [word for word in prompt if word not in stopwords.words('english')]
        prompt = ''.join([f'{word} ' for word in prompt])

        output = []
        for batch in range(BATCH_NUM):
            prompt_vec = np.zeros(len(VOC[batch]))
            for word in prompt.split():
                if word in VOC[batch]:
                    prompt_vec[VOC[batch][word]] += 1
            prompt_vec = prompt_vec / np.linalg.norm(prompt_vec)
            scores = np.dot(US[batch], V[batch] @ prompt_vec)
            scores = np.array([scores[i] for i in range(len(scores)) if TITLES[batch][i] != prompt])
            scores = np.argsort(scores)[::-1]
            
            for i in range(10):
                output.append((scores[i], TITLES[batch][scores[i]]))
            output = sorted(output, key=lambda x: x[0], reverse=True)
            output = output[:10]
        
        print("Results:")
        for i in range(10):
            print(f"{i + 1}. Acc: {output[i][0]}, Title: {output[i][1]}")

    else:
        print("Bye!")
        break
            

