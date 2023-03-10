import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error



df = pd.DataFrame({'name': ['Raphael', 'Donatello'],
                 'mask': ['red', 'purple'],
                'weapon': ['sai', 'bo staff']})

#print(df)
#df.to_csv('out.csv', index=False)

test = pd.read_csv(r'.\test.csv')
a = np.array([sum(test.loc[i][1:11]) for i in range(len(test))])
b = len(test.loc[0][1:11])
print(y_predicted - a/b)