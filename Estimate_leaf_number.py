# Author: Fanhang Zhang
# Time: 2024/4/9 13:01

import joblib
import pandas as pd

# load model
model = joblib.load('XXX.pkl')  # model path

# load data
file_path = 'input_data_XXX.txt'  # input_file path
df = pd.read_csv(file_path, sep='\t', index_col=0)
X = df.iloc[:, :]

# estimate leaf number
y_pred = model.predict(X)

df_pred = pd.DataFrame(y_pred, columns=['Estimated leaf number'], index=df.index)

df_pred.to_csv('output_data_XXX.txt', sep='\t')  # ouput_file path
