# This file contains the main code to create the ML model, load the data, 
# https://www.youtube.com/watch?v=0B5eIE_1vpU 54:00
# imports
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder
import matplotlib.pylab as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklego.meta import GroupedPredictor #WARNING: check how balanced groups are bfore use
import pandas as pd


def load_data(file_path: str):
	"""
	Reads input data from specified csv file
	Args:
		file_path (str): _description_

	Returns:
		X (pd.DataFrame): samples matrix (design matrix)
		y (pd.Series): target values
	"""
	# read data into pandas dataframe
	df = pd.read_csv(file_path, parse_dates=["date"])
	# transform data into all numerical values
	df = pd.get_dummies(df) # one-hot encode columns "type" and "geography"
	df["date"] = df["date"].dt.month # convert dates into numerical representation of months i.e. 1:12
	# TODO: do date by cyclic function like sin
	# TODO: check how year is scaled: should it just be years since start instead?
	X = df.loc[:, df.columns!="average_price"]
	# X = df.loc[:, df.columns!="average_price"].iloc[:,:9]
	y = df["average_price"]
	# add test that post processing all data in x is numeric
	return X, y


def evaluate_model(pred,y):
	plt.scatter(pred, y)
	plt.show()
	print(f"accuracy: {str(mean_absolute_error(y, pred))}")


def grid_search(pipe, X, y):
	mod = GridSearchCV(estimator=pipe,
                 param_grid={
                   'model__n_neighbors': list(range(10,20))
                 },
                 cv=3) 
	mod.fit(X,y)
	pd.DataFrame(mod.cv_results_).to_excel("grid_search_results.xlsx")
	evaluate_model(mod.predict(X),y)


def train_test(pipe, X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	pipe.fit(X_train, y_train)
	evaluate_model(pipe.predict(X_test),y_test)
	

if __name__ == "__main__":
	X, y = load_data("avocado-data.csv")
	# scaler = StandardScaler() # Quantile was slightly better
	scaler = QuantileTransformer(n_quantiles=100)
	# scaler = PolynomialFeatures() # bad
	# model = LogisticRegression()
	# model = LinearRegression() # linear regression did poorly
	model = KNeighborsRegressor(n_neighbors=10)
	pipe = Pipeline([
		("scale", scaler),
		("model", model)
	])

	pred=train_test(pipe,X,y)
	pred=grid_search(pipe,X,y)