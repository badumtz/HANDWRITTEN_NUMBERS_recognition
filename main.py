from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



digits = datasets.load_digits()

dir(digits)



def plot_multi(i):
    nplots= 16
    fig = plt.figure(figsize=(13,13))
    for j in range(nplots):
        plt.subplot(4, 4, j+1)
        plt.imshow(digits.images[i+j], cmap='binary')
        plt.title(digits.target[i+j])
        plt.axis('off')

plot_multi(0)

y= digits.target
x= digits.images.reshape((len(digits.images), -1))

x_train= x[:1000]
y_train= y[:1000]

x_test= x[1000:]
y_test= y[1000:]

mlp = MLPClassifier(hidden_layer_sizes=(13,),
                    activation='logistic',
                    alpha=1e-4, solver='sgd',
                    tol=1e-4, random_state=1,
                    learning_rate_init=.1,
                    verbose=True)


mlp.fit(x_train, y_train)
predictions= mlp.predict(x_test)

ACCURACY = accuracy_score(y_test, predictions)
print(ACCURACY)