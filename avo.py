# This file contains the main code to create the ML model, load the data, 

# imports
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, \
								  QuantileTransformer, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklego.meta import GroupedPredictor #WARNING: check how balanced groups are bfore use


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def week_of_month(day):
	"""
	Determine which week in a month a given day falls under
	Args:
		day (int in 1:31): day of the month

	Returns:
		week_num (int in 0:4): week of the month
	"""
	# assert day>=1 & day<=31, f"day {day} is not in valid range"
	return int(np.ceil((day-1)//7))
	
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
	# perform feature engineering and transform input data into numeric values
	df = pd.get_dummies(df) # one-hot encode columns "type" and "geography"
	df["month"] = df["date"].dt.month # extract numerical representation of months (i.e. 1:12) from dates
	df["day"] = df["date"].dt.day # extract numerical representation of days (i.e. 1:31) from dates
	df["week_of_month"]=df["day"].apply(week_of_month) 
	# convert months into cyclical data in range -1:1
	# df["month_cyclical"] = np.cos(2 * np.pi * df["month"]/11.0) 
	df["month_cyclical"] = np.sin(2 * np.pi * df["month"]/11.0)
	df = df.drop("date",axis=1) # no longer need raw date data after extracting desired features
	X = df.loc[:, df.columns!="average_price"]
	# X = df.loc[:, df.columns!="average_price"].iloc[:,:9]
	y = df["average_price"]
	# TODO add test that post processing all data in x is numeric
	return X, y

def evaluate_model(pred,y):
	plt.plot
	plt.scatter(pred, y)
	plt.show()
	print(f"mean squared error: {str(mean_squared_error(y, pred))}")
	print(f"mean absolute error: {str(mean_absolute_error(y, pred))}")


def search(pipe, X, y):
	mod = GridSearchCV(estimator=pipe,
                 param_grid={
                   'model__n_neighbors': list(range(10,20))
                 },
                 cv=3) 
	mod.fit(X,y)
	pd.DataFrame(mod.cv_results_).to_excel("search_results.xlsx")
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
	pred=search(pipe,X,y)