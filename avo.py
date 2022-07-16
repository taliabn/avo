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

def evaluate_model(mod: GridSearchCV):
	pred = mod.fit(X, y).predict(X)
	plt.scatter(pred, y)
	plt.show()
	pd.DataFrame(mod.cv_results_).to_excel("grid_search_results.xlsx")
	return("my metric of choice")

if __name__ == "__main__":
	X, y = load_data("avocado-data.csv")
	# scaler = StandardScaler() # Quantile was slightly better
	scaler = QuantileTransformer(n_quantiles=100)
	# scaler = PolynomialFeatures() # bad
	# model = LogisticRegression()
	# model = LinearRegression() # linear regression did poorly
	model = KNeighborsRegressor(n_neighbors=10)
	enc = OneHotEncoder(handle_unknown="ignore")
	pipe = Pipeline([
		("scale", scaler),
		("model", model)
	])

	mod = GridSearchCV(estimator=pipe,
                 param_grid={
                   'model__n_neighbors': list(range(10,20))
                 },
                 cv=2) 
	evaluate_model(mod)