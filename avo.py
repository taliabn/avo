# This file contains the main code to create the ML model, load the data, 
# https://www.youtube.com/watch?v=0B5eIE_1vpU 54:00
# imports
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder
import matplotlib.pylab as plt
from sklearn.model_selection import GridSearchCV
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
	df = pd.read_csv(file_path, parse_dates=["date"])
	df["date"] = df["date"].dt.month # convert dates into numerical representation of months i.e. 1:12
	# TODO: do date by cyclic function like sin
	# TODO: https://stackoverflow.com/questions/60153981/scikit-learn-one-hot-encoding-certain-columns-of-a-pandas-dataframe
	# X = df.loc[:, df.columns!="average_price"]
	X = df.loc[:, df.columns!="average_price"].iloc[:,:9]
	y = df["average_price"]
	return X, y


if __name__ == "__main__":
	# scaler = StandardScaler() # Quantile was slightly better
	scaler = QuantileTransformer(n_quantiles=100)
	# scaler = PolynomialFeatures() # bad
	# model = LogisticRegression()
	# model = LinearRegression() # linear regression did poorly
	model = KNeighborsRegressor(n_neighbors=10)
	enc = OneHotEncoder(handle_unknown="ignore")
	X, y = load_data("avocado-data.csv")
	pipe = Pipeline([
		("scale", scaler),
		("model", model)
	])
	pred = pipe.fit(X, y).predict(X)
	plt.scatter(pred, y)
	plt.show()
	# mod = GridSearchCV(estimator=pipe,
    #              param_grid={
    #                'model__n_neighbors': list(range(10,20))
    #              },
    #              cv=3) 
	# mod.fit(X, y)
	
	# pd.DataFrame(mod.cv_results_).to_clipboard