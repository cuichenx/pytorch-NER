import torch
import numpy as np
from utils import regex_parsable, hmong_syllable_component
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

run_name = 'grpd2_letters_chardim8_Clf_run1'
print('run_name is', run_name)
model_path = f"model/{run_name}/best_model"
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



