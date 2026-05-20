import pandas as pd    
jsonObj = pd.read_json(path_or_buf="DeepJSONEval.jsonl", lines=True)

jsonObj.to_json("data.json", orient="records", indent=4)

#import json as json_module

#with open("data.json", "w", encoding="utf-8") as f:
 #   json_module.dump(jsonObj, f, indent=4)