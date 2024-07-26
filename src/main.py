from sklearn import datasets
from sklearn.model_selection import train_test_split
from gradient_descent import LinearRegression as LinRegGD
from QR_decomposition import LinearRegression as LinRegQR
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

def main():

	X, y = datasets.load_boston(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

	# Linear regression solved by Gradient Descent algorithm
	linreg_GD = LinRegGD()
	linreg_GD.fit(X_train, y_train)

	# Linear regression solved by QR-decomposition
	linreg_QR = LinRegQR()
	linreg_QR.fit(X_train, y_train)

	# Standard sklearn implementation
	linreg_lib = linear_model.LinearRegression()
	linreg_lib.fit(X_train, y_train)

	# show results
	plt.figure()
	plt.subplot(1, 3, 1)
	report("Gradient Descent", linreg_GD, X_test, y_test)
	plt.subplot(1, 3, 2)
	report("QR-decomposition", linreg_QR, X_test, y_test)
	plt.subplot(1, 3, 3)
	report("Sklearn LinearReg", linreg_lib, X_test, y_test)
	plt.show()

def report(label, model, X, y):
	y_pred = model.predict(X)
	plt.scatter(x=y, y=y_pred, label=label, alpha=0.5)
	plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', c='red', lw=1)
	plt.title("Predicted vs. Actual")
	plt.xlabel("Actual")
	plt.ylabel("Predictions")
	plt.legend()	
	mse = np.mean( (y - y_pred)**2 )
	r2 = 1 - np.sum( (y-y_pred)**2 ) / np.sum( (y-np.mean(y))**2 )
	print(f"{label: <20} mse={mse:.4f}     r2={r2:.5f}")
	        
if __name__ == '__main__':
	main()
