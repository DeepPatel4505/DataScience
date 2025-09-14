from sklearn.datasets import make_regression
from matplotlib import pyplot as plt

x, y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20, random_state=43)

plt.scatter(x, y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

from models import SimpleGD

model = SimpleGD(learning_rate=0.01, epochs=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score

print("SimpleGD R2:", r2_score(y_test, y_pred))

from sklearn.linear_model import SGDRegressor

mdl = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant', penalty=None, random_state=43)
mdl.fit(x_train, y_train)

ypred = mdl.predict(x_test)
print("SGDRegressor R2:", r2_score(y_test, ypred))

print("SGDRegressor coef, intercept:", mdl.coef_, mdl.intercept_)