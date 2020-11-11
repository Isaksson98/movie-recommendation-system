import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix


df = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/filis220.training", header=None)
#each row in dara: u,m,r: 

user_id = df[df.columns[0]]
movie_id = df[df.columns[1]]
ratings = df[df.columns[2]]

avg_rating= ratings.mean() #3.6

#min //x-At//^2 regards to t

x = ratings-avg_rating
movies = [i+2000 for i in movie_id]
users = [i for i in user_id]

def create_matrix_A():
    #print(df.index.values)

    dw = df.index.values
    arr = np.concatenate((df.index.values, dw))

    row  = np.array(arr)
    c=users+movies
    col  = np.array(c)
    g = len(ratings)
    dat = [1 for i in c]

    data=np.array(dat)

    num = 177735*2+1;

    sparse_matrix = coo_matrix((data, (row, col)), shape=(num, num)) #2000+1500 from pdf

    print(sparse_matrix.tocsr()[:,6])

    ans = np.linalg.lstsq(sparse_matrix, y, rcond=None)[0]

create_matrix_A()