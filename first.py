import numpy as np
from sklearn.datasets import make_blobs
#X, y = make_blobs(centers=2, random_state=0)

X_train = np.array([[5, 1, 2, 4], [1, 4, 5, 2], [4, 2, 2, 4],[4, 2, 1, 5],[2, 5, 4, 1], [5, 1, 3, 5], [2, 4, 4, 1], [4, 1, 1, 4],[5, 2, 1, 4],[1, 5, 5, 2]], np.int32)
y_train = np.array([[1], [2], [1], [1], [2], [1], [2], [1],[1],[2]], np.int32)

#shy(5 for very shy), talkative(5 for very talkative), group of unknown ppl(1 for not at all comfortable), alone(5 for comfortable/happy being alone)

X_test = np.array([[2, 2, 3, 3],], np.int32)
y_test = np.array([[2]], np.int32)

from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
print(prediction)
print(y_test)
np.mean(prediction == y_test)
