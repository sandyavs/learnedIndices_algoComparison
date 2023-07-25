import json

resultFileName = "learned_index_results.json"
#you can specify a  path, you can put another extension to the file but is best to put .pickle to now what is each file
def read_json_file():
	with open(resultFileName, "r") as openFile:
		data = json.load(openFile)
	return data

def modify_json_data(jsonObj, algo, datasetName, total_build_time, total_query_time):
	if datasetName == 'D1' or 'D2' or 'D3':
		jsonObj[datasetName][algo]['buildTime'] = total_build_time
		jsonObj[datasetName][algo]['queryTime'] = total_query_time
	else:
		print('Invalid Dataset input')
	return jsonObj

#saving data
def write_json_file(jsonObj):
	with open(resultFileName, "w") as writeFile:
		# pickle.dump(jsonObj, writeFile)
		json.dump(jsonObj, writeFile, ensure_ascii=False, indent=4)
		return
