import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
#from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error



#each row in dara: u,m,r: 

#train_data = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/filis220.training", header=None)
#test_data = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/filis220.test", header=None)

#task2_test_data = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/task---2.test", header=None)
#task2_train_data = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/task---2.training", header=None)

verification_test_data = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/verification.test", header=None)
verification_train_data = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/verification.training", header=None)



def create_matrix_A(train_set):
    
    user_ids = np.array(train_set[ train_set.columns[0]])
    movie_ids = np.array(train_set[ train_set.columns[1]])
    ratings = np.array(train_set[ train_set.columns[2]])

    users = [i for i in user_ids]

    user_len = max(users)

    movies = [j+user_len for j in movie_ids]

    col_len=users+movies
    movie_len= max(movies)

    dw = train_set.index.values
    arr = np.concatenate((dw, dw))

    row  = np.array(arr) #just index 0,1,2...
    col  = np.array(col_len)
    
    dat = [1 for i in col_len]
    data=np.array(dat)


    sparse_matrix = csr_matrix((data, (row, col-1)), shape=(len(ratings), movie_len))

    return sparse_matrix

A = create_matrix_A(verification_train_data)
def task1(train_set, test_set):

    user_ids = np.array(train_set[ train_set.columns[0]])
    movie_ids = np.array(train_set[ train_set.columns[1]])
    ratings = np.array(train_set[ train_set.columns[2]])

    test_user_ids = np.array(test_set[ test_set.columns[0]])
    test_movie_ids = np.array(test_set[ test_set.columns[1]])
    test_ratings = np.array(test_set[ test_set.columns[2]])

    avg_rating = ratings.mean()

    x = ratings-avg_rating

    ans = np.linalg.lstsq(A.todense(), x, rcond=None)[0] #optimal b*
    b_user = ans[:max(test_user_ids)] #first part of vector
    b_movies = ans[-max(test_movie_ids):] #second part of vector
    
    r_prediction = []
    for i in range(len(test_ratings)):
        r_prediction.append( [avg_rating+b_user[test_user_ids[i]-1]+b_movies[test_movie_ids[i]-1]])

    
    r_prediction_arr=np.array(r_prediction) #baseline prediction rating
    lowerBound=1
    upperBound=5
    np.clip(r_prediction_arr, lowerBound, upperBound, out=r_prediction_arr)


    print(r_prediction_arr.shape)
    print(test_ratings.shape)
    print(ratings.shape)

    RMSE_test = np.sqrt(np.mean((r_prediction_arr-test_ratings)**2))
    #diff_train = r_prediction_arr-ratings
    #RMSE verification.training is 0.891 and for verification.test is 0.905.
    print('RMSE:')
    print(RMSE_test)
    rms = math.sqrt(mean_squared_error(test_ratings, r_prediction_arr))
    print(rms)
    #RMSE_train = np.sqrt(diff_train**2)

    #H1 = np.rint(RMSE_test).astype(int)
    #H2 = np.rint(RMSE_train).astype(int)

    #print(RMSE_train.mean())


    #plt.subplot(2,2,1)
    #plt.hist(H1, bins=[1,2,3,4,5])  # `density=False` would make counts
    #plt.ylabel('Probability')
    #plt.xlabel('Data');
    #plt.title('Test')

    #plt.subplot(2,2,2)
    #plt.hist(H2, bins=5)  # `density=False` would make counts
    #plt.ylabel('Probability')
    #plt.xlabel('Data');
    #plt.title('Train')
    #plt.show()
    print('plot')

    b_user2 = ans[:max(user_ids)] #first part of vector
    b_movies2 = ans[-max(movie_ids):] #second part of vector

    r_prediction2 = []
    for i in range(len(ratings)):
        r_prediction2.append( [avg_rating+b_user2[user_ids[i]-1]+b_movies2[movie_ids[i]-1]])

    
    target=np.array(r_prediction2) #baseline prediction rating


    lowerBound=1
    upperBound=5
    np.clip(target, lowerBound, upperBound, out=target)


    print(target.shape)
    print(test_ratings.shape)
    print(ratings.shape)

    rms2 = math.sqrt(mean_squared_error(ratings, target))
    #RMSE_train = np.sqrt(np.mean((target-ratings)*(target-ratings)))
    
    #RMSE verification.training is 0.891 and for verification.test is 0.905.
    print('RMSE:')
    #print(RMSE_train)
    print(rms2)
    return r_prediction_arr

r_prediction_arr = task1(verification_train_data, verification_test_data)

###TASK 2###
print('task2')
user_id_t2 = train_data[task2_train_data.columns[0]]
movie_id_t2 = train_data[task2_train_data.columns[1]]
ratings_t2 = train_data[task2_train_data.columns[2]]

#r_pred = r_prediction_arr.round(decimals=3).T
#r_tilde = ratings_t2 - r_pred 

def cosSimilarity(movie1_id, movie2_id):
    x=0
    temp1 = 0
    temp2 = 0
    temp3 = 0

    for i in range(len(user_id_t2)):
        temp1 += ratings_t2[movie1_id]*ratings_t2[movie2_id]
        temp2 += ratings_t2[movie1_id]**2
        temp3 += ratings_t2[movie2_id]**2

    x = math.sqrt(temp1)/(math.sqrt(temp2)*math.sqrt(temp3))
    return x

print(cosSimilarity(2,5))
print(cosSimilarity(3,6))
print(cosSimilarity(4,7))
print(cosSimilarity(5,8))
print(cosSimilarity(6,9))
print(cosSimilarity(9,9))