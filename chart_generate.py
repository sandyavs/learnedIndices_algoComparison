import pandas as pd
import matplotlib.pyplot as plot
from inputFileProcess import *

objects = [read_json_file()]
# with (open(resultFileName, "rb")) as openfile:
# 	while True:
# 		try:
# 			objects.append(pickle.load(openfile))
# 		except EOFError:
# 			break

bPlusTree = {
	"buildTime": [],
	"queryTime": []
}
symbolic_reg =  {
	"buildTime": [],
	"queryTime": []
}
spn =   {
	"buildTime": [],
	"queryTime": []
}

for dct in objects:
	for key, inner_dct in dct.items():
		print(f"Outer Key: {key}")
		for inner_key, inner_val in inner_dct.items():
			if inner_key == 'bPlusTree':
				bPlusTree["buildTime"].append(inner_val['buildTime'])
				bPlusTree["queryTime"].append(inner_val['queryTime'])
			elif inner_key == 'SR':
				symbolic_reg["buildTime"].append(inner_val['buildTime'])
				symbolic_reg["queryTime"].append(inner_val['queryTime'])
			elif inner_key == 'SPN':
				spn["buildTime"].append(inner_val['buildTime'])
				spn["queryTime"].append(inner_val['queryTime'])
			else:
				print('Err in data mapping')
			print(f"Inner Key: {inner_key}, Values: {inner_val}")
print(bPlusTree, symbolic_reg, spn)



index = ["Dataset1 (10000)", "Dataset2 (100k)", "Dataset3 (1M)"]

data_build_time = {
	"BPlusTree": bPlusTree["buildTime"],
	"Symbolic Regression": symbolic_reg["buildTime"],
	"SPN": spn["buildTime"]
}
dataFrame = pd.DataFrame(data=data_build_time, index=index)
# Draw a vertical bar chart
dataFrame.plot.bar(rot=15, title="Build time comparision for 3 algorithms")
plot.show(block=True)
plot.savefig('buildIndex.png')

data_query_time = {
	"BPlusTree": bPlusTree["queryTime"],
	"Symbolic Regression": symbolic_reg["queryTime"],
	"SPN": spn["queryTime"]
}
# Dictionary loaded into a DataFrame
dataFrame = pd.DataFrame(data=data_query_time, index=index)
# Draw a vertical bar chart
dataFrame.plot.bar(rot=15, title="Query time comparision for 3 algorithms")
plot.show(block=True)
plot.savefig('queryPlot.png')
