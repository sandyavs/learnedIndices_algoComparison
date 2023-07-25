import random  # for self test
import argparse  # for CLI test
import time  # timing index building and query time
from datetime import datetime  # For logging
import os  # For current filename
from inputFileProcess import *

"""
In-Memory B+ Tree
"""

__author__ = "benben233 https://github.com/benben233"
__version__ = "unknown"
__license__ = "unspecified"

splits = 0
parent_splits = 0
fusions = 0
parent_fusions = 0


class Node(object):
	"""Base node object. It should be index node
	Each node stores keys and children.

	Attributes:
		parent
	"""

	def __init__(self, parent=None):
		"""Child nodes are stored in values. Parent nodes simply act as a medium to traverse the tree.
		:type parent: Node"""
		self.keys: list = []
		self.values: list[Node] = []
		self.parent: Node = parent

	def index(self, key):
		"""Return the index where the key should be.
		:type key: str
		"""
		for i, item in enumerate(self.keys):
			if key < item:
				return i

		return len(self.keys)

	def __getitem__(self, item):
		return self.values[self.index(item)]

	def __setitem__(self, key, value):
		i = self.index(key)
		self.keys[i:i] = [key]
		self.values.pop(i)
		self.values[i:i] = value

	def split(self):
		"""Splits the node into two and stores them as child nodes.
		extract a pivot from the child to be inserted into the keys of the parent.
		@:return key and two children
		"""
		global splits, parent_splits
		splits += 1
		parent_splits += 1

		left = Node(self.parent)

		mid = len(self.keys) // 2

		left.keys = self.keys[:mid]
		left.values = self.values[:mid + 1]
		for child in left.values:
			child.parent = left

		key = self.keys[mid]
		self.keys = self.keys[mid + 1:]
		self.values = self.values[mid + 1:]

		return key, [left, self]

	def __delitem__(self, key):
		i = self.index(key)
		del self.values[i]
		if i < len(self.keys):
			del self.keys[i]
		else:
			del self.keys[i - 1]

	def fusion(self):
		global fusions, parent_fusions
		fusions += 1
		parent_fusions += 1

		index = self.parent.index(self.keys[0])
		# merge this node with the next node
		if index < len(self.parent.keys):
			next_node: Node = self.parent.values[index + 1]
			next_node.keys[0:0] = self.keys + [self.parent.keys[index]]
			for child in self.values:
				child.parent = next_node
			next_node.values[0:0] = self.values
		else:  # If self is the last node, merge with prev
			prev: Node = self.parent.values[-2]
			prev.keys += [self.parent.keys[-1]] + self.keys
			for child in self.values:
				child.parent = prev
			prev.values += self.values

	def borrow_key(self, minimum: int):
		index = self.parent.index(self.keys[0])
		if index < len(self.parent.keys):
			next_node: Node = self.parent.values[index + 1]
			if len(next_node.keys) > minimum:
				self.keys += [self.parent.keys[index]]

				borrow_node = next_node.values.pop(0)
				borrow_node.parent = self
				self.values += [borrow_node]
				self.parent.keys[index] = next_node.keys.pop(0)
				return True
		elif index != 0:
			prev: Node = self.parent.values[index - 1]
			if len(prev.keys) > minimum:
				self.keys[0:0] = [self.parent.keys[index - 1]]

				borrow_node = prev.values.pop()
				borrow_node.parent = self
				self.values[0:0] = [borrow_node]
				self.parent.keys[index - 1] = prev.keys.pop()
				return True

		return False


class Leaf(Node):
	def __init__(self, parent=None, prev_node=None, next_node=None):
		"""
		Create a new leaf in the leaf link
		:type prev_node: Leaf
		:type next_node: Leaf
		"""
		super(Leaf, self).__init__(parent)
		self.next: Leaf = next_node
		if next_node is not None:
			next_node.prev = self
		self.prev: Leaf = prev_node
		if prev_node is not None:
			prev_node.next = self

	def __getitem__(self, item):
		return self.values[self.keys.index(item)]

	def __setitem__(self, key, value):
		i = self.index(key)
		if key not in self.keys:
			self.keys[i:i] = [key]
			self.values[i:i] = [value]
		else:
			self.values[i - 1] = value

	def split(self):
		global splits
		splits += 1

		left = Leaf(self.parent, self.prev, self)
		mid = len(self.keys) // 2

		left.keys = self.keys[:mid]
		left.values = self.values[:mid]

		self.keys: list = self.keys[mid:]
		self.values: list = self.values[mid:]

		# When the leaf node is split, set the parent key to the left-most key of the right child node.
		return self.keys[0], [left, self]

	def __delitem__(self, key):
		i = self.keys.index(key)
		del self.keys[i]
		del self.values[i]

	def fusion(self):
		global fusions
		fusions += 1

		if self.next is not None and self.next.parent == self.parent:
			self.next.keys[0:0] = self.keys
			self.next.values[0:0] = self.values
		else:
			self.prev.keys += self.keys
			self.prev.values += self.values

		if self.next is not None:
			self.next.prev = self.prev
		if self.prev is not None:
			self.prev.next = self.next

	def borrow_key(self, minimum: int):
		index = self.parent.index(self.keys[0])
		if index < len(self.parent.keys) and len(self.next.keys) > minimum:
			self.keys += [self.next.keys.pop(0)]
			self.values += [self.next.values.pop(0)]
			self.parent.keys[index] = self.next.keys[0]
			return True
		elif index != 0 and len(self.prev.keys) > minimum:
			self.keys[0:0] = [self.prev.keys.pop()]
			self.values[0:0] = [self.prev.values.pop()]
			self.parent.keys[index - 1] = self.keys[0]
			return True

		return False


class BPlusTree(object):
	"""B+ tree object, consisting of nodes.

	Nodes will automatically be split into two once it is full. When a split occurs, a key will
	'float' upwards and be inserted into the parent node to act as a pivot.

	Attributes:
		maximum (int): The maximum number of keys each node can hold.
	"""
	root: Node

	def __init__(self, maximum=4):
		self.root = Leaf()
		self.maximum: int = maximum if maximum > 2 else 2
		self.minimum: int = self.maximum // 2
		self.depth = 0

	def find(self, key) -> Leaf:
		""" find the leaf

		Returns:
			Leaf: the leaf which should have the key
		"""
		node = self.root
		# Traverse tree until leaf node is reached.
		while type(node) is not Leaf:
			node = node[key]

		return node

	def __getitem__(self, item):
		return self.find(item)[item]

	def query(self, key):
		"""Returns a value for a given key, and None if the key does not exist."""
		leaf = self.find(key)
		return leaf[key] if key in leaf.keys else None

	def change(self, key, value):
		"""change the value

		Returns:
			(bool,Leaf): the leaf where the key is. return False if the key does not exist
		"""
		leaf = self.find(key)
		if key not in leaf.keys:
			return False, leaf
		else:
			leaf[key] = value
			return True, leaf

	def __setitem__(self, key, value, leaf=None):
		"""Inserts a key-value pair after traversing to a leaf node. If the leaf node is full, split
			the leaf node into two.
			"""
		if leaf is None:
			leaf = self.find(key)
		leaf[key] = value
		if len(leaf.keys) > self.maximum:
			self.insert_index(*leaf.split())

	def insert(self, key, value):
		"""
		Returns:
			(bool,Leaf): the leaf where the key is inserted. return False if already has same key
		"""
		leaf = self.find(key)
		if key in leaf.keys:
			return False, leaf
		else:
			self.__setitem__(key, value, leaf)
			return True, leaf

	def insert_index(self, key, values: list[Node]):
		"""For a parent and child node,
					Insert the values from the child into the values of the parent."""
		parent = values[1].parent
		if parent is None:
			values[0].parent = values[1].parent = self.root = Node()
			self.depth += 1
			self.root.keys = [key]
			self.root.values = values
			return

		parent[key] = values
		# If the node is full, split the  node into two.
		if len(parent.keys) > self.maximum:
			self.insert_index(*parent.split())
		# Once a leaf node is split, it consists of a internal node and two leaf nodes.
		# These need to be re-inserted back into the tree.

	def delete(self, key, node: Node = None):
		if node is None:
			node = self.find(key)
		del node[key]

		if len(node.keys) < self.minimum:
			if node == self.root:
				if len(self.root.keys) == 0 and len(self.root.values) > 0:
					self.root = self.root.values[0]
					self.root.parent = None
					self.depth -= 1
				return

			elif not node.borrow_key(self.minimum):
				node.fusion()
				self.delete(key, node.parent)
		# Change the left-most key in node
		# if i == 0:
		#     node = self
		#     while i == 0:
		#         if node.parent is None:
		#             if len(node.keys) > 0 and node.keys[0] == key:
		#                 node.keys[0] = self.keys[0]
		#             return
		#         node = node.parent
		#         i = node.index(key)
		#
		#     node.keys[i - 1] = self.keys[0]

	def show(self, node=None, file=None, _prefix="", _last=True):
		"""Prints the keys at each level."""
		if node is None:
			node = self.root
		print(_prefix, "`- " if _last else "|- ", node.keys, sep="", file=file)
		_prefix += "   " if _last else "|  "

		if type(node) is Node:
			# Recursively print the key of child nodes (if these exist).
			for i, child in enumerate(node.values):
				_last = (i == len(node.values) - 1)
				self.show(child, file, _prefix, _last)

	def output(self):
		return splits, parent_splits, fusions, parent_fusions, self.depth

	def readfile(self, reader):
		i = 0
		for i, line in enumerate(reader):
			s = line.decode().split(maxsplit=1)
			self[s[0]] = s[1]
			if i % 1000 == 0:
				print('Insert ' + str(i) + 'items')
		return i + 1

	def leftmost_leaf(self) -> Leaf:
		node = self.root
		while type(node) is not Leaf:
			node = node.values[0]
		return node


# Self test
def demo():
	bplustree = BPlusTree()
	random_list = random.sample(range(1, 100), 20)
	for i in random_list:
		bplustree[i] = 'test' + str(i)
		print('Insert ' + str(i))
		bplustree.show()

	random.shuffle(random_list)
	for i in random_list:
		print('Delete ' + str(i))
		bplustree.delete(i)
		bplustree.show()


if __name__ == '__main__':
	# demo() # Self test

	# %% Setup command line arguments

	# Ref: https://github.com/eriknyquist/duckargs
	parser = argparse.ArgumentParser(description='',
									formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# Input text file representing a sorted column of ints as 1st positional argument
	parser.add_argument('input_sorted_int_file', type=argparse.FileType(
		'r'), help='Specify the URI of sorted integer text file')

	# Specify the number of times index to be built
	parser.add_argument('-b', '--build-times', default=10,
						type=int, help='No. of times index to be built')

	# Specify the number of times a query may be run for a key that exists
	parser.add_argument('-s', '--query-success-times', default=10, type=int,
						help='No. of times a query may be run for a key that exists')

	# Specify the number of times a query may be run for a key that does not exist
	parser.add_argument('-n', '--query-null-times', default=10, type=int,
						help='No. of times a query may be run for a key that does NOT exist')

	# Specify whether we need verbose output
	parser.add_argument('-v', '--verbose', default=False,
						type=bool, help='flag for verbose output')

	# Specify output of "--version"
	parser.add_argument(
		"--version",
		action="version",
		version="%(prog)s (version {version})".format(version=__version__)
	)

	version = "%(prog)s (version {version})".format(version=__version__)
	args = parser.parse_args()

	# %% Begin processing
	print("Input Dataset: D1/D2/D3 ?")
	datasetName = input()

	# Extract CLI arguments
	infile_URI = args.input_sorted_int_file.name
	isVerbose = args.verbose
	build_times = args.build_times
	query_success_times = args.query_success_times
	query_null_times = args.query_null_times

	# TODO: Positivity checks for inputs

	# %% Initialise timekeeping variables
	csv_to_list_runtime = [None] * 2
	build_index_runtime = [None] * 2
	query_success_runtime = [None] * 2
	query_null_runtime = [None] * 2

	# WIP CSV output
	print("Log_tag, index, input, task, metric_label, value")

	# %% Read a single cloumn text CSV as a list of ints
	if isVerbose:
		print("Input file: ", infile_URI)

	csv_to_list_runtime[0] = time.time()

	# Read as text
	with open(infile_URI, "r") as text:
		lines = text.read().splitlines()

	# Convert str list to int
	column_data_as_list = [int(x) for x in lines]
	# print("column_data_as_list", len(column_data_as_list))
	csv_to_list_runtime[1] = time.time()

	# if isVerbose:
	#     print("column_data: ", column_data_as_list)

	# Report read time
	current_task = "read_csv_to_list"
	current_metric = "runtime"
	now = datetime.now()
	current_time = now.strftime("[%Y%m%d:%H%M%S]")
	print(current_time, "[LOG]", ", ",
		os.path.basename(__file__), ", ",
		os.path.basename(infile_URI), ", ",
		current_task, ", ",
		current_metric, ", ",
		csv_to_list_runtime[1] - csv_to_list_runtime[0],
		sep="")

	# %% Build indices

	build_start_time = time.time()
	for no_iter in [*range(0, build_times, 1)]:
		build_index_runtime[0] = time.time()

		# Create new bplus tree
		bplustree = BPlusTree(maximum=2)

		# Insert all column data
		for column_data_pos in [*range(0, len(column_data_as_list), 1)]:
			bplustree.insert(
				column_data_as_list[column_data_pos], column_data_pos)

		# if isVerbose:
		#     print("B-plus tree: ")
		#     bplustree.show()

		build_index_runtime[1] = time.time()

		current_task = "build_index"
		current_metric = "runtime"
		now = datetime.now()
		current_time = now.strftime("[%Y%m%d:%H%M%S]")
		# print(current_time,"[LOG]", ", ",
		#       os.path.basename(__file__), ", ",
		#       os.path.basename(infile_URI), ", ",
		#       current_task, ", ",
		#       current_metric, ", ",
		#       build_index_runtime[1] - build_index_runtime[0],
		#       sep = "")

		current_metric = "index_size"
		# TODO: Consider deeper inspection of actual memory usage,
		# Ref: https://pythonhosted.org/Pympler/classtracker.html#classtracker
	build_end_time = time.time()
	total_build_time = round((build_end_time - build_start_time) * 1000, 2)
	print("Total build time: for", build_times, "builds", total_build_time, " ms")

	# %% Query index for existing key
	current_task = "query_success"
	query_start_time = time.time()
	for no_iter in [*range(0, query_success_times, 1)]:

		query_point = random.sample(column_data_as_list,  1)[0]

		# if isVerbose:
		#     print("query: ", query_point)

		# query_success_runtime[0] = time.time()

		# result is absolute position of the integer key
		query_result = bplustree.query(query_point)

		# if isVerbose:
		#     print("query result: ", query_result)

		# query_success_runtime[1] = time.time()

		# current_task = "query_success"
		# current_metric = "runtime"
		# now = datetime.now()
		# current_time = now.strftime("[%Y%m%d:%H%M%S]")
		# print(current_time,"[LOG]", ", ",
		#       os.path.basename(__file__), ", ",
		#       os.path.basename(infile_URI), ", ",
		#       current_task, ", ",
		#       current_metric, ", ",
		#       query_success_runtime[1] - query_success_runtime[0],
		#       sep = "")
	query_end_time = time.time()
	total_query_time = round((query_end_time - query_start_time) * 1000, 2)
	print("Total query execution time: for", query_success_times, "queries", total_query_time, " ms")
	avg_query_exec_time = total_query_time/query_success_times
	print("Avg query execution time: ", f"{avg_query_exec_time: .2f}")

	# %% Query index for non-existing key
	# print("column_data_as_list", max(column_data_as_list))
	# Some value that definitely does not exist
	query_point = max(column_data_as_list) + 1
	# if isVerbose:
	#     print("query: ", query_point)

	for no_iter in [*range(0, query_null_times, 1)]:

		query_null_runtime[0] = time.time()

		# result is absolute position of the integer key
		query_result = bplustree.query(query_point)

		if isVerbose:
			print("query result: ", query_result)

		query_null_runtime[1] = time.time()

		current_task = "query_null"
		current_metric = "runtime"
		now = datetime.now()
		current_time = now.strftime("[%Y%m%d:%H%M%S]")
		print(current_time, "[LOG]", ", ",
			os.path.basename(__file__), ", ",
			os.path.basename(infile_URI), ", ",
			current_task, ", ",
			current_metric, ", ",
			query_null_runtime[1] - query_null_runtime[0],
			sep="")

	jsonObj = read_json_file()
	jsonObj = modify_json_data(read_json_file(), 'bPlusTree', datasetName, total_build_time, total_query_time)
	write_json_file(jsonObj)