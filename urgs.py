import random
import re
import time

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from memory_profiler import profile
from inputFileProcess import *

np.seterr(all='raise')


def integer_to_binary(number: int) -> str:
	"""
	Converts an integer to a binary string
	:param number: The integer to convert
	:return: The binary string
	"""

	# noinspection StrFormat
	return f'{{:0>16}}'.format(bin(number)[2:])


def list_input_variables():
	"""
	Reads the input variables from the csv file
	:return: A list of the input variables
	"""

	return pd.read_csv(path_to_csv).keys()[:-1].tolist()


def list_parameters():
	"""
	Creates a list of the parameters
	:return: A list of the parameters
	"""
	parameters = []
	# noinspection PyShadowingNames
	for i in range(0, symbolic_regression_parameters["maximum_fitting_parameters"]):
		parameters.append('p' + str(i))
	return parameters


def get_bits(binary: str):
	""""
	Splits a binary string into its odd and even bits
	:param binary: The binary string
	"""

	even_bits = binary[::2]
	odd_bits = binary[1::2]
	return odd_bits, even_bits


def get_left_right_bits(integer: int):
	"""
	Splits an integer into its odd and even bits or left and right bits
	:param integer: The integer to split
	:return: The left and right integers
	"""

	left_bin, right_bin = get_bits(integer_to_binary(integer))
	left_int = int(left_bin, 2)
	right_int = int(right_bin, 2)
	return left_int, right_int


def get_dataset_properties(dataframe, variable):
	"""
	Returns the mean, standard deviation, minimum and maximum of a variable in a dataframe
	:param dataframe: The dataframe to calculate the properties from
	:param variable: The variable to get the properties of
	:return: A dictionary of the properties of the variable
	"""
	properties = dict()
	properties[variable + '_mean'] = dataframe.mean()
	properties[variable + '_std'] = dataframe.std()
	properties[variable + '_min'] = dataframe.min()
	properties[variable + '_max'] = dataframe.max()

	return properties


def exponential_notation(number: float) -> str | None:
	"""
	The number 1,000,000 can be written in scientific notation as 1 x 10^6.
	Eg for this function: 1000000.0 would be returned as 1.00E+06

	:param number: The number to be converted to scientific notation
	:return: The number in scientific notation
	"""

	formatted = "{:.2E}".format(number).replace(
		"E", " x 10 ^ ").replace("+", "").replace("00", "0")
	if formatted == "1.0 x 10 ^ 0":
		return None
	return formatted


def sin(x):
	return np.sin(x)


def cos(x):
	return np.cos(x)


def tan(x):
	return np.tan(x)


def log(x):
	return np.log(np.abs(x))


def sinh(x):
	return np.sinh(x)


def cosh(x):
	return np.cosh(x)


def tanh(x):
	return np.tanh(x)


# noinspection PyShadowingBuiltins
def sum(x):
	return np.sum(x)


def add(x, y):
	return np.add(x, y)


def sub(x, y):
	return np.subtract(x, y)


def mul(x, y):
	return np.multiply(x, y)


def div(x, y):
	return np.divide(x, y)


# noinspection PyShadowingBuiltins
def pow(x, y):
	return np.power(x, y)


def sqrt(x):
	return np.sqrt(x)


# noinspection PyShadowingBuiltins
def abs(x):
	return np.abs(x)


def neg(x):
	return np.negative(x)


def get_element_of_cartesian_product(*args, repeat=1, index=0):
	"""
	You want the i^th index of the cartesian product of two large arrays?
	This function retrieves that element without needing to iterate through
	the entire product
	"""
	pools = [tuple(pool) for pool in args] * repeat
	if len(pools) == 0:
		return []
	len_product = len(pools[0])
	len_pools = len(pools)
	for j in range(1, len_pools):
		len_product *= len(pools[j])
	if index >= len_product:
		raise Exception("index + 1 is bigger than the length of the product")
	index_list = []
	for j in range(0, len_pools):
		ith_pool_index = index
		denom = 1
		for k in range(j + 1, len_pools):
			denom *= len(pools[k])
		ith_pool_index //= denom
		if j != 0:
			ith_pool_index %= len(pools[j])
		index_list.append(ith_pool_index)
	ith_item = []
	for index in range(0, len_pools):
		ith_item.append(pools[index][index_list[index]])
	return ith_item


# noinspection PyProtectedMember,PyPep8Naming,PyShadowingNames
def equation_generator(enumerator):
	"""
	Generates the random equation
	i is in the domain [0, N-1], and specifies which binary tree to use
	q is in the domain [0, G-1], and specifies which function of arity 1 configuration
	r is in the domain [0, A-1], and specifies which function of arity 2 configuration
	s is in the domain [0, B-1], and specifies which terminal configuration
	"""

	# Generates a random equation given the number of permitted unique trees
	# a Dataset object, and an Enumerator object
	#
	# It works by generating the random numbers which specify the equation,
	# then passing those as arguments to equation_generator
	max_trees = 100
	n = len(symbolic_regression_parameters["arity_2"])
	f = len(symbolic_regression_parameters["arity_1"])
	m = dataset.m_terminals
	i = random.choices(range(0, max_trees))[0]
	q = enumerator.get_q(f, i)
	r = enumerator.get_r(n, i)
	s = enumerator.get_s(m, i)

	tree = plant_binary_tree(i)
	G = enumerator.get_G(f, i)
	if q >= G and not G == 0:
		raise Exception("q is an index that must be smaller than G")
	A = enumerator.get_A(n, i)
	if r >= A:
		raise Exception("r is an index that must be smaller than A")
	# noinspection PyPep8Naming
	B = enumerator.get_B(m, i)
	if s >= B:
		raise Exception("s is an index that must be smaller than B")
	l_i = enumerator.get_l_i(i)
	k_i = enumerator.get_k_i(i)
	j_i = enumerator.get_j_i(i)
	# of all the possible configurations of arity 1 functions, we pick the
	# configuration at index q
	f_func_config = get_element_of_cartesian_product(symbolic_regression_parameters["arity_1"],
													repeat=l_i, index=q)
	# of all the possible configurations of arity 2 functions, we pick the
	# configuration at index r
	n_func_config = get_element_of_cartesian_product(symbolic_regression_parameters["arity_2"],
													repeat=k_i, index=r)
	# of all the possible configurations of terminals, we pick the
	# configuration at index s
	term_config = get_element_of_cartesian_product(dataset.terminals_list,
												repeat=j_i, index=s)
	# the trees are generated in the form [. , .] where . denotes a leaf,
	# and the square brackets indicate a function
	# we do some string replacements here, according to the determined
	# configurations to build the equation as a string
	for z in range(0, len(n_func_config)):
		func = n_func_config[z]
		tree = tree.replace('[', func + '(', 1)
		tree = tree.replace(']', ')', 1)
	for z in range(0, len(f_func_config)):
		func = f_func_config[z]
		tree = tree.replace('|', func + '(', 1)
		tree = tree.replace('|', ')', 1)
	for z in range(0, len(term_config)):
		term = term_config[z]
		tree = tree.replace('.', term, 1)
	if '|' in tree:
		return None
	return tree


class Dataset:
	"""
	This class is used to load the data from the csv file
	and to store the data along with the labels and properties
	"""

	def __init__(self):
		(dataframe, header_labels) = Dataset.load_csv_data()
		self.maximum_fitting_parameters = symbolic_regression_parameters[
			"maximum_fitting_parameters"]
		self.dataframe = dataframe
		self.header_labels = header_labels
		self.x_data, self.x_labels, x_properties = self.get_input_variables()
		self.y_data, self.y_label, y_properties = self.get_output_variable()
		if np.std(self.y_data) == 0:
			raise Exception("The data is invalid. All y values are the same.")
		self.properties = dict()
		self.properties.update(x_properties)
		self.properties.update(y_properties)
		self.data_dictionary = self.get_data_dictionary()
		self.no_of_input_variables = len(self.x_labels)
		self.m_terminals = self.no_of_input_variables + self.maximum_fitting_parameters
		self.terminals_list = (list_parameters() + list_input_variables())

	# noinspection PyShadowingNames
	@staticmethod
	def load_csv_data():
		"""
		Loads the data from the csv file
		:return: Dataframe and the header labels
		"""
		# use only for hundredK dataset
		# dataframe = pd.read_csv(path_to_csv)

		dataframe = pd.read_csv(path_to_csv, header=None)

		# use only to generate index col
		# temp_y = [*range(1, len(dataframe[0:])+1, 1)]
		# dataframe["y"] = temp_y

		column_labels = dataframe.keys()
		column_labels = ["x", "y"]
		dataframe = dataframe.set_axis(column_labels, axis=1)
		print(dataframe.head)

		# Switch x and y columns, i.e. 0th and 1st column, @author: Avin
		col_list = list(dataframe)
		col_list[0], col_list[1] = col_list[1], col_list[0]
		dataframe.columns = col_list
		return dataframe, column_labels

	def get_input_variables(self):
		"""
		Gets the input variables and their labels and  properties from the dataframe
		:return: Input variables, their labels and properties
		"""

		# get all the columns except the last one
		features = self.dataframe.iloc[:, :-1]
		features = np.array(features)
		labels = self.header_labels[:-1]
		properties = dict()
		for label in labels:
			properties.update(get_dataset_properties(
				self.dataframe[label], label))

		return features, labels, properties

	def get_output_variable(self):
		"""
		Gets the output variable and its label and properties from the dataframe
		:return: Output variable, its label and properties
		"""

		# get the last column
		feature = self.dataframe.iloc[:, -1]
		feature = np.array(feature)
		label = self.header_labels[-1]
		properties = get_dataset_properties(self.dataframe[label], label)

		return feature, label, properties

	def get_data_dictionary(self):
		"""
		Returns the data in the csv file as a dictionary for use in eva()
		"""

		data_dictionary = dict()
		for label in self.header_labels:
			data_dictionary[label] = np.array(
				self.dataframe[label].values).astype(float)
			if np.any(np.isnan(data_dictionary[label])):
				raise Exception("The data is invalid. There are NaN values.")

		return data_dictionary


class Enumerator:
	"""
	Enumerates the problem space and counts all possible equations
	Section 2.3 of the paper
	"""

	# noinspection PyPep8Naming,PyShadowingNames
	def get_G(self, f, i):
		# G is the number of ways to pick l_i functions of arity
		# one from f possible functions of arity one
		l_i = self.get_l_i(i)
		G = pow(f, l_i)
		G = int(G)
		return G

	# noinspection PyPep8Naming,PyShadowingNames
	def get_A(self, n, i):
		# A is the number of ways to pick k_i functions of arity
		# two from n possible functions of arity two
		k = self.get_k_i(i)
		A = pow(n, k)
		A = int(A)
		return A

	# noinspection PyPep8Naming,PyShadowingNames
	def get_B(self, m, i):
		# B is the number of ways to pick j_i terminals from m terminals
		j = self.get_j_i(i)
		B = pow(m, j)
		B = int(B)
		return B

	# noinspection PyShadowingNames,PyPep8Naming
	def get_q(self, f, i):
		G = self.get_G(f, i)
		q = random.randint(0, G - 1)
		return q

	# noinspection PyPep8Naming,PyShadowingNames
	def get_r(self, n, i):
		A = self.get_A(n, i)
		r = random.randint(0, A - 1)
		return r

	# noinspection PyPep8Naming,PyShadowingNames
	def get_s(self, m, i):
		B = self.get_B(m, i)
		s = random.randint(0, B - 1)
		return s

	# noinspection PyShadowingNames
	def get_l_i(self, i):
		i = int(i)
		# from n functions of arity two, pick k_i
		# k_i is the number of non-leaf nodes in the tree corresponding to i
		if i == 0:
			l_i = 0
		elif i == 1:
			l_i = 1
		elif i == 2:
			l_i = 0
		else:
			left_int, right_int = get_left_right_bits(i)
			left_l_i = self.get_l_i(left_int)
			right_l_i = self.get_l_i(right_int)
			l_i = left_l_i + right_l_i + 1
		return l_i

	# noinspection PyShadowingNames
	def get_k_i(self, i):
		i = int(i)
		# from n functions of arity two, pick k_i
		# k_i is the number of non-leaf nodes in the tree corresponding to i
		if i == 0:
			k_i = 0
		elif i == 1:
			k_i = 1
		elif i == 2:
			k_i = 0
		else:
			left_int, right_int = get_left_right_bits(i)
			left_k_i = self.get_k_i(left_int)
			right_k_i = self.get_k_i(right_int)
			k_i = left_k_i + right_k_i + 1
		return k_i

	# noinspection PyShadowingNames
	def get_j_i(self, i):
		i = int(i)
		# from m m_terminals, pick j_i
		# j_i is the number of leafs in the tree corresponding to i
		if i == 0:
			j_i = 1
		elif i == 1:
			j_i = 2
		elif i == 2:
			j_i = 1
		else:
			left_int, right_int = get_left_right_bits(i)
			left_j_i = self.get_j_i(left_int)
			right_j_i = self.get_j_i(right_int)
			j_i = left_j_i + right_j_i
		return j_i


def create_fitting_parameters(max_parameters, parameter_values=None) -> lmfit.Parameters:
	"""
	Creates a lmfit.Parameters object with max_parameters parameters
	:param max_parameters: The number of parameters to create
	:param parameter_values: Must be None or a np.array of length max_parameters
	:return: lmfit.Parameters object
	"""

	parameters = lmfit.Parameters()
	for parameter_index in range(0, max_parameters):
		name = 'p' + str(parameter_index)
		param_init_value = float(1)
		parameters.add(name, param_init_value)
	if parameter_values is not None:
		for parameter_index in range(0, max_parameters):
			name = 'p' + str(parameter_index)
			parameters[name].value = parameter_values[parameter_index]

	return parameters


# noinspection PyProtectedMember,PyUnusedLocal,PyUnboundLocalVariable
def evaluate_equation(parameters: lmfit.Parameters, function_string, mode='residual'):
	"""
	Evaluates the equation given by function_string and returns the output
	Uses the eval function to evaluate the equation
	:param parameters: Parameters object
	:param function_string: String representation of the symbolic equation to be evaluated
	:param mode: 'residual' or 'y_calculation' or a dictionary
	:return: result of the equation
	"""

	no_of_records = len(dataset.y_data)
	df = dataset.data_dictionary
	pd = parameters.valuesdict()
	y_label = dataset.y_label
	output_vector = df[y_label]
	residual = [float('inf')] * len(df[y_label])
	# Used to calculate the residual between the predicted values and the actual values of the target variable.
	if mode == 'residual':
		eval_string = '(' + function_string + ') -  df["' + y_label + '"]'
		residual = eval(eval_string)
		output = residual
	# Used to calculate the predicted values of the target variable.
	elif mode == 'y_calculation':
		y_value = eval(function_string)
		output = y_value
	elif type(mode) == dict:
		df = mode
		try:
			y_value = eval(function_string)
		except KeyError:
			print(function_string)
		output = y_value
	# if model only has parameters and no data variables, we can have a
	# situation where output is a single constant
	# noinspection PyUnboundLocalVariable
	if np.size(output) == 1:
		output = np.resize(output, np.size(output_vector))

	return output


# noinspection PyShadowingBuiltins
def add_dictionary_lookup_for_eval(equation_string: str) -> str:
	"""
	Surrounds keys in the equation string with the dictionary variable to create a dictionary lookup expression so that
	it can be evaluated by eval().

	:param equation_string: The equation string to be evaluated by eval() in terms of x and p0, p1, etc.
	:return: The equation string with x replaced by df["x"] and p0, p1, etc. replaced by params["p0"], params["p1"],
	etc.
	"""

	list_of_inputs = re.findall(r'x\d+', equation_string)
	if len(list_of_inputs) == 1:
		equation_string = equation_string.replace(
			list_of_inputs[0], 'df["' + list_of_inputs[0] + '"]')
	elif len(list_of_inputs) == 0:
		equation_string = equation_string.replace('x', 'df["x"]')
	else:
		for input in set(list_of_inputs):
			equation_string = equation_string.replace(
				input, 'df["' + input + '"]')
	list_of_parameters = re.findall(r'p\d+', equation_string)
	for parameter in set(list_of_parameters):
		equation_string = equation_string.replace(
			parameter, 'parameters["' + parameter + '"]')

	return equation_string


# noinspection PyUnresolvedReferences
def check_goodness_of_fit(equation, parameters):
	"""
	Evaluates the goodness of fit between a given mathematical equation and a dataset,
	using the specified input parameter values.
	:param equation: The equation to be evaluated
	:param parameters: The parameters to be used in the evaluation
	"""

	function_string = add_dictionary_lookup_for_eval(str(equation))

	# residual is the difference between the actual y values of the dataset and the predicted y values
	# lower the residual, better the fit
	if len(parameters) > 0:
		# minimize the residual of the equation with respect to the parameters
		result = lmfit.minimize(evaluate_equation, parameters, args=[function_string],
								nan_policy='propagate')
		residual = result.residual
		y_calculation = evaluate_equation(
			result.params, function_string, mode='y_calculation')
		parameters_dictionary_to_store = result.params
	else:
		residual = evaluate_equation(parameters, function_string)
		y_calculation = evaluate_equation(
			parameters, function_string, mode='y_calculation')
		parameters_dictionary_to_store = parameters

	# average_y_data is the average of the actual y values of the dataset
	average_y_data = np.average(dataset.y_data)
	# sum of squared residuals is the sum of the squared differences between the actual y values
	# and the predicted y values
	sum_of_squared_residuals = sum(pow(residual, 2))
	# sum of squared totals is the sum of the squared differences between the actual y values and the average y value
	sum_of_squared_totals = sum(pow(y_calculation - average_y_data, 2))
	# coefficient of determination meaning how much of the variation in the target variable is explained by the model
	r_square = 1 - sum_of_squared_residuals / sum_of_squared_totals

	return sum_of_squared_residuals, sum_of_squared_totals, r_square, parameters_dictionary_to_store, residual


# noinspection PyShadowingNames
def plant_binary_tree(i):
	"""
	Returns the binary tree

	:param i: The index of the binary tree
	:return: the tree duh
	"""
	# empty sibling for arity 1 functions
	if i == 0:
		tree = '.'
	# for arity 2 functions
	elif i == 1:
		tree = '[., .]'
	# for arity 1 functions
	elif i == 2:
		tree = '|.|'
	else:
		left_int, right_int = get_left_right_bits(i)
		left = plant_binary_tree(left_int)
		right = plant_binary_tree(right_int)
		tree = '[' + left + ', ' + right + ']'
	return tree


class ResultList:
	"""
	A list of results from Symbolic Regression.
	"""

	def __init__(self):
		self.results: list[Result] = []

	# noinspection PyProtectedMember
	def sort(self) -> None:
		"""
		Using only mean square error makes it more accurate but hard to read
		Using only equation length makes it easier to read but less accurate
		Therefore, using a combination by scaling the mean square error by 15
		"""
		self.results = sorted(self.results,
							key=lambda x:
							len(x.equation)
							+
							x.mean_square_error * 15
							)

	def print(self):
		"""
		Prints the results in a nice format
		"""

		rows = []
		column_names = ["Normalized MSE", "R**2", "Model", "Parameters"]
		row = self.results[0].attributes()
		rows.append(row)
		for j in range(len(rows)):
			for k in range(len(rows[j])):
				print(column_names[k] + ': ' + str(rows[j][k]))


class Result:
	"""
	Class to save a single result of SRURGS
	"""

	def __init__(self, equation, mean_square_error, r_square, parameter):
		self.equation = equation
		self.mean_square_error = mean_square_error
		self.r_square = r_square
		self.parameters = np.array(parameter)

	def attributes(self):
		"""
		Returns a list of all attributes of the result
		"""

		summary = [self.mean_square_error, self.r_square, self.equation]
		parameters = []
		for parameter in self.parameters:
			formatted_parameter = exponential_notation(parameter)
			if formatted_parameter is not None:
				parameters.append(formatted_parameter)
		parameters_str = ', '.join(parameters)
		summary.append(parameters_str)

		return summary


# noinspection PyShadowingNames,PyPep8Naming,PyProtectedMember,PyUnboundLocalVariable
def srurgs(results):
	"""
	Runs Symbolic Regression using Uniform Random Global Search (SRURGS)
	:param results: The dictionary of results
	"""

	enumerator = Enumerator()
	valid = False
	while not valid:
		equation = equation_generator(enumerator)
		if equation is None:
			raise ValueError("Equation is None")
		try:
			parameters = create_fitting_parameters(5)
			(sum_of_squared_residuals, sum_of_squared_totals, r_square, parameters_fitted,
			residual) = check_goodness_of_fit(equation, parameters)
			valid = True
		except FloatingPointError:
			pass
	nmse = sum_of_squared_residuals / np.std(dataset.y_data)
	result = Result(equation, nmse, r_square, parameters_fitted)

	results[equation] = result


# noinspection PyShadowingNames
def get_resultlist(results) -> ResultList:
	"""
	Converts the dictionary of results to a ResultList object
	:param results:
	:return:
	"""

	result_list = ResultList()

	for equation in results.keys():
		result = results[equation]
		result_list.results.append(result)
	return result_list


# noinspection PyShadowingNames,PyPep8Naming,PyProtectedMember
def display(results: dict) -> None:
	"""
	Sorts the results to find the best model and prints it

	:param results: Dictionary with all results
	"""
	result_list = get_resultlist(results)
	result_list.sort()
	result_list.print()


# noinspection PyShadowingNames,PyPep8Naming,PyProtectedMember
# def plot_results(results):
# 	"""
# 	Plots the results of the best model
# 	:param results: Dictionary with all results
# 	"""

# 	if len(dataset.x_labels) != 1:
# 		print("\n**Can only plot 1D data**")
# 		return
# 	result_list = get_resultlist(results)
# 	result_list.sort()
# 	best_model = result_list.results[0]
# 	parameters = create_fitting_parameters(symbolic_regression_parameters['maximum_fitting_parameters'],
# 										parameter_values=best_model.parameters)
# 	equation_string = add_dictionary_lookup_for_eval(best_model.equation)
# 	data_dictionary = dict()
# 	xlabel = dataset.x_labels[0]
# 	# data_dictionary[xlabel] = dataset._x_data
# 	# create an array of values for the input variable using np.linspace() and add it to the data dictionary
# 	data_dictionary[xlabel] = np.linspace(np.min(dataset.x_data),
# 										np.max(dataset.x_data))
# 	# noinspection PyTypeChecker
# 	# predict the output values using the best model
# 	y_calculated = evaluate_equation(
# 		parameters, equation_string, mode=data_dictionary)
# 	plt.plot(data_dictionary[xlabel], y_calculated,
# 			'b-', label=dataset.y_label + ' calculated')
# 	plt.plot(dataset.x_data, dataset.y_data, 'ro',
# 			label=dataset.y_label + ' original data')
# 	plt.xlabel(dataset.x_labels[0])
# 	plt.ylabel(dataset.y_label)
# 	plt.title('Best model')
# 	plt.legend()
# 	plt.show()


def timeit(method):
	"""
	Decorator to time a function
	:param method: The function to time
	"""

	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()

		print(method.__name__, te - ts)
		return result

	return timed


# noinspection PyShadowingNames
@profile
# @timeit
def run(attempts: int, results: dict):
	"""
	Runs SRURGS a number of times
	:param attempts: Number of times to run SRURGS
	:param results: Dictionary to store results
	:return:
	"""
	for i in range(0, attempts):
		try:
			srurgs(results)
		except ValueError:
			continue


# noinspection PyShadowingNames
def write_results(results):
	open('results.csv', 'w').close()
	file = open('results.csv', 'a')
	file.write('x,y\n')
	result_list = get_resultlist(results)
	result_list.sort()
	best_model = result_list.results[0]
	eq = best_model.equation
	for i in range(len(best_model.parameters)):
		eq = eq.replace('p' + str(i), str(best_model.parameters[i]))
	for i in range(10_00_000):
		new_eq = eq.replace('x', str(i / 100000))
		file.write(str(i / 100000) + ',' + str((eval(new_eq))) + '\n')
	file.close()


# noinspection PyShadowingNames
@profile
# @timeit
def predict(x: int | list, results):
	result_list = get_resultlist(results)
	result_list.sort()
	best_model = result_list.results[0]
	if type(x) == int:
		eq = best_model.equation
		eq = eq.replace('x', str(x))
		for i in range(len(best_model.parameters)):
			eq = eq.replace('p' + str(i), str(best_model.parameters[i]))
		result = int(eval(eq))
		return result
	elif type(x) == list:
		y = np.zeros((len(x), 2))
		for i in range(len(x)):
			y[i] = predict(x[i], results)

		return y
	else:
		raise TypeError("x must be a int or list of ints")


def plot_points():
	"""
	Plots the points from the results.csv file
	"""
	file = open('results.csv', 'r')
	lines = file.readlines()
	x = []
	y = []
	for line in lines[1:]:
		line = line.split(',')
		x.append(line[0])
		y.append(line[1])
	plt.plot(x, y, 'ro')
	plt.show()


if __name__ == '__main__':
	# Configuration
	# path_to_csv = './csvs/hundredK.csv'
	print("Input Dataset: D1/D2/D3 ?")
	datasetName = input()
	path_to_csv = './dataset_csv/int_1000000_20230717_123143_train_aifeynman.ssv'
	attempts = 100  # no of times the algo runs

	# Define the nature of the search space in a symbolic regression problem
	symbolic_regression_parameters = {
		'maximum_fitting_parameters': 3,
		'arity_2': ['add', 'sub', 'mul', 'div', 'pow'],
		'arity_1': ['log', 'sinh', 'sin', 'cosh', 'cos', 'tanh', 'sqrt', 'abs', 'neg']
		# 'arity_1': ['log', 'sinh', 'sin', 'cosh', 'cos', 'tanh', 'tan', 'sqrt', 'abs', 'neg'],
	}

	results = dict()
	dataset = Dataset()

	# Run SRURGS multiple times
	start_time = time.time()
	start_build_time = time.time()
	run(attempts, results)
	end_build_time = time.time()
	total_build_time = round((end_build_time - start_build_time) * 1000, 2) #in ms

	# Display results and plot best model
	# display(results)
	# plot_results(results)
	# predict(49_999, results)
	# print(predict(49_999, results))
	# print(predict([1148863700450000, 1530000], results))
	# input("Press Enter to continue...")
	# write_results(results)
	# plot_points()

	# Compare predictions to actual positions
	temp_x = dataset.dataframe['x']
	temp_y = dataset.dataframe['y']
	dataset_x_list = pd.Series.tolist(temp_x)
	dataset_y_list = pd.Series.tolist(temp_y)
	
	noOfQueries = 300
	pairs = list(zip(dataset_x_list, dataset_y_list))  # make pairs out of the two lists
	pairs = random.sample(pairs, noOfQueries)  # pick random pairs
	temp_x_list, temp_y_list = zip(*pairs)  # separate pairs
	temp_x_list = list(temp_x_list)
	temp_y_list = list(temp_y_list)

	start_predict_time = time.time()
	pred_y = predict(temp_x_list, results)
	pred_y_list = [pred_y[x][0] for x in [*range(0, len(temp_y_list), 1)]]
	end_predict_time = time.time()
	total_prediction_time = round((end_predict_time - start_predict_time) * 1000, 2)  # in milliseconds
	# print("[DEBUG][urgs] Actual positions: ", temp_y_list)
	# print("[DEBUG][urgs] Predicted positions: ", pred_y_list)

	print("Total Build time: ", f"{total_build_time: .2f}", " ms")
	print("Total query execution time for", noOfQueries, "queries:", f"{total_prediction_time: .2f}", " ms")
	avg_prediction_time = total_prediction_time/noOfQueries
	print("Avg query execution time: ", f"{avg_prediction_time: .2f}", " ms")

	jsonObj = read_json_file()
	jsonObj = modify_json_data(read_json_file(), 'SR', datasetName, total_build_time, total_prediction_time)
	write_json_file(jsonObj)
	

	# plt.scatter(temp_y_list, pred_y_list)
	# # plt.plot( [*range(0, len(temp_y_list), 1)], [*range(0, len(temp_y_list), 1)], c="red")
	# plt.plot( [*range(0, len(temp_y_list), 1)], [*range(0, len(temp_y_list), 1)], c="red")
	# plt.title('SURG based learned index')
	# plt.xlabel("Actual position")
	# plt.ylabel("Predicted position")
	# plt.show()
