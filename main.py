import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

train = pd.read_csv(r'.\train.csv')
test = pd.read_csv(r'.\test.csv')


#print(train)
#train.y

#print(len(train))
phi = np.array([train.loc[i][2:12] for i in range(len(train))])
phi_T = phi.transpose()

#Inverse = np.linalg.inv(np.matmul(phi_T, phi))

w_hat = np.dot(np.linalg.inv(np.matmul(phi_T, phi)), np.dot(phi_T, train.y))

# test testset

phi_test = np.array([test.loc[i][1:11] for i in range(len(test))])

y_predicted = (np.dot(phi_test, w_hat)).transpose()
print(y_predicted)

df = pd.DataFrame({'id': [int(test.loc[i][0]) for i in range(len(test))],
                   'y': y_predicted})
df.to_csv('consignement.csv', index=False)