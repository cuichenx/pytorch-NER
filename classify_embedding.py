import os

import torch
import numpy as np
from utils import regex_parsable, hmong_syllable_component
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def get_model_path(run_name):
    SERVER = "/run/user/1000/gvfs/sftp:host={}.lti.cs.cmu.edu,user=cxcui/usr0/home/cxcui/pytorch-NER/model/"
    for server_name in ('agent', 'patient'):
        server_base = SERVER.format(server_name)
        if not os.path.exists(server_base):
            print(f"WARNING: {server_name} not mounted")
        path = os.path.join(server_base, run_name, 'best_model')
        if os.path.exists(path):
            return path
    print(run_name, "not found!")

run_name = 'grpd1_nochar_noswap_tr0.10_run3'
print('run_name is', run_name)
model_path = get_model_path(run_name)
embeds = torch.load(model_path, map_location='cpu')['embeds.weight']


data_path = "data/data.pth"
data = torch.load(data_path)
i2w = data['i2w']

valid_wi = []
y_tones = []  # ton labels
tones = '_bdgjmsv'
tone2i = {'': 0, 'b': 1, 'd': 2, 'g': 3, 'j': 4, 'm': 5, 's': 6, 'v': 7}
i2tone = ['', 'b', 'd', 'g', 'j', 'm', 's', 'v']

for i, w in enumerate(i2w):
    if regex_parsable(w):
        valid_wi.append(i)
        y_tones.append(hmong_syllable_component(w, 'ton'))

X = embeds[np.array(valid_wi)]
y = list(map(tone2i.get, y_tones))
print(len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# clf = SVC()
clf = MLPClassifier(hidden_layer_sizes=(100, 50, 20), verbose=True, early_stopping=True, tol=1e-4, n_iter_no_change=20)

clf.fit(X_train, y_train)
train_acc = clf.score(X_train, y_train)
print('train accuracy is', train_acc)
test_acc = clf.score(X_test, y_test)
print('test accuracy is', test_acc)



