import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
#from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import math



train_data = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/filis220.training", header=None)
#each row in dara: u,m,r: 

test_data = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/filis220.test", header=None)

user_id = train_data[train_data.columns[0]]
movie_id = train_data[train_data.columns[1]]
ratings = train_data[train_data.columns[2]]

avg_rating= ratings.mean() #3.6

user_id_test = test_data[test_data.columns[0]]
movie_id_test = test_data[test_data.columns[1]]
ratings_test = test_data[test_data.columns[2]]

x = ratings-avg_rating
movies = [i+2000 for i in movie_id]
users = [i for i in user_id]


def create_matrix_A():

    dw = train_data.index.values
    arr = np.concatenate((train_data.index.values, dw))

    row  = np.array(arr)
    c=users+movies
    col  = np.array(c)
    g = len(ratings)
    dat = [1 for i in c]

    data=np.array(dat)

    num = 177735*2+1;

    #sparse_matrix = coo_matrix((data, (row, col)), shape=(num, num)) #2000+1500 from pdf
    sparse_matrix = csr_matrix((data, (row, col)), shape=(177735, 3501))

    return sparse_matrix

A = create_matrix_A()

ans = np.linalg.lstsq(A.todense(), x, rcond=None)[0]
b_user=ans[:2000]
b_movies=ans[1500:]

r_prediction = []
for i in range(len(ratings_test)):
    r_prediction.append( [avg_rating+b_user[user_id_test[i]-1]+b_movies[movie_id_test[i]-1]])


r_prediction_arr=np.array(r_prediction)
ratings_test_arr=np.array(ratings_test)


diff=r_prediction_arr-ratings_test_arr
RMSE = np.sqrt(diff**2)
H = np.round(RMSE)
print(H)
plt.hist(H, bins=5)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Data');
plt.show()

print( r_prediction_arr.mean() )
