# Import Excel file -> pd.read_excel
#import pandas as pd
# df = pd.read_excel(io=input_filepath, sheet_name=sheet)

# Import txt file -> read_csv
#import pandas as pd
#df = pd.read_csv(input_filepath, sep=',')

# Import csv file -> read_csv
#import pandas as pd
#df = pd.read_csv(input_filepath, sep=',')

# Open a Python object
import cPickle
def open_anything(path_to_open):
    with open(path_to_open, 'rb') as f:
        object_opened = cPickle.load(f)
    return object_opened

# Save a Python object
import cPickle
def save_anything(object_to_save, path_for_save):
    with open(path_for_save, 'wb') as f:
        cPickle.dump(object_to_save, f)

# Read a JSON file
import json
def read_json(path_to_open):
    jsonFile = json.loads(open(path_to_open).read())
    return jsonFile

# Save a JSON file
import json
def save_to_json(object_to_save, path_for_save):
    with open(path_for_save, 'w+') as outfile:
        json.dump(object_to_save, outfile)

# Import parquet dataset (to dataFrame)
import pandas as pd
import pyarrow.parquet as pq
def read_parquet_to_df(path_to_open):
    data_parquet = pq.read_table(path_to_open, nthreads=4)
    df = data_parquet.to_pandas()
    return df

# Save DataFrame to csv file
import pandas as pd
def save_df_to_csv(df_to_save, path_to_save, header=True,sep=";"):
    df_to_save.to_csv(path_to_save, header=header, index=False, sep=sep, decimal=".")


# Import a json dataset
import json
import pandas as pd
def read_json_dataset(path_to_open):
    data = []
    with open(path_to_open) as f:
        for line in f:
            jsonData = json.loads(line)
            data.append(jsonData)
    data = pd.DataFrame.from_dict(data, orient='columns')
    return data


# More complicated, nested json
import json
import pandas as pd
def read_json_dataset_with_apps(config):
    data = []
    dataApps = []
    with open(config['local_path_raw_data']) as f:
        for line in f:
            jsonData = json.loads(line)
            if "apps" in config['categorical_dimensions']:
                try:
                    dataApps.append(jsonData["apps"])
                except:
                    dataApps.append({})
                        
            try:
                del jsonData["apps"]
            except:
                pass
            data.append(jsonData)
    
    df = pd.DataFrame.from_dict(data, orient='columns')
    print(df.shape)
       
    if "apps" in config['categorical_dimensions']:
        dfApps = pd.DataFrame.from_dict(dataApps, orient='columns')
        dfApps = dfApps.fillna(0)
        print(dfApps.shape)
        dfApps.columns = 'apps__' + dfApps.columns
        df = df.join(dfApps)
    
    print(df.shape)
    return df
