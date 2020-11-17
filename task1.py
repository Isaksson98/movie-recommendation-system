import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
#from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from scipy import spatial



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


    vect = test_ratings-r_prediction_arr
    vect2 = np.square(vect)
    RMSE_test = np.sqrt(np.mean(vect2))
    #diff_train = r_prediction_arr-ratings
    #RMSE verification.training is 0.891 and for verification.test is 0.905.
    print('RMSE:')
    #print(RMSE_test)
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


    b_user2 = ans[:max(user_ids)] #first part of vector
    b_movies2 = ans[-max(movie_ids):] #second part of vector

    r_prediction2 = []
    for i in range(len(ratings)):
        r_prediction2.append( [avg_rating+b_user2[user_ids[i]-1]+b_movies2[movie_ids[i]-1]])

    
    target=np.array(r_prediction2) #baseline prediction rating


    lowerBound=1
    upperBound=5
    np.clip(target, lowerBound, upperBound, out=target)


    rms2 = math.sqrt(mean_squared_error(ratings, target))
    #RMSE_train = np.sqrt(np.mean((target-ratings)*(target-ratings)))
    
    #RMSE verification.training is 0.891 and for verification.test is 0.905.
    print('RMSE:')
    #print(RMSE_train)
    print(rms2)
    return r_prediction_arr

baseline_prediction = task1(verification_train_data, verification_test_data)

###TASK 2###
def task2(train_set, test_set, baseline_prediction):
    user_ids = np.array(train_set[ train_set.columns[0]])
    movie_ids = np.array(train_set[ train_set.columns[1]])
    ratings = np.array(train_set[ train_set.columns[2]])

    test_user_ids = np.array(test_set[ test_set.columns[0]])
    test_movie_ids = np.array(test_set[ test_set.columns[1]])
    test_ratings = np.array(test_set[ test_set.columns[2]])
    print('')

    r_tilde_ = np.subtract(test_ratings, np.transpose(baseline_prediction))
    r_tilde = r_tilde_[0]
    
    print('shapes: ')
    print(test_ratings.shape)
    print(r_tilde.shape)
    
    new_prediction = []
    m = 5
    L = 100
    for i in range(L):
       
        cos_sim = cosSimilarity(m, movie_ids[i], user_ids, movie_ids, ratings, train_set)[0]
        num1 = cos_sim * r_tilde[i]
        num2 = abs(cos_sim)
        num3 = num1/num2
        new_prediction.append(num3)

    prediction=np.array(new_prediction) #improved prediction rating
    
    print(ratings.shape)
    print(prediction.shape)

    rms = math.sqrt(mean_squared_error(ratings, prediction))

    return rms


def cosSimilarity(movie1_id, movie2_id, users, movies, ratings, train_set):
    x = 0
    temp1 = 0
    temp2 = 0
    temp3 = 0

    ids_movie1 = np.asarray(np.where( (movies == movie1_id) )) #Index of all movies 1
    ids_movie2 = np.asarray(np.where( (movies == movie2_id) )) #index of all movies 2

    users_movies1 = users[ids_movie1] #Users that rated film1
    users_movies2 = users[ids_movie2] #Users that rated film2
    
    ratings_mov1 = ratings[ids_movie1]
    ratings_mov2 = ratings[ids_movie2]

    num4 = np.asarray(np.intersect1d(users_movies1, users_movies2)) #List of users that rated film 1 & 2

    rating_index1 = []
    rating_index2 = []

    for i in ids_movie1[0]:

        if (users[i] in num4):
            rating_index1.append(ratings[i])

    for k in ids_movie2[0]:
        if users[k] in num4:
            rating_index2.append(ratings[k])

    for j in range(len(num4)):
        
        temp1 += rating_index1[j]*rating_index2[j]
        temp2 += rating_index1[j]**2
        temp3 += rating_index2[j]**2

    x = temp1/(math.sqrt(temp2)*math.sqrt(temp3))
    result = 1 - spatial.distance.cosine(rating_index1, rating_index2)

    return [x,result]

final_prediction = task2(verification_train_data, verification_test_data, baseline_prediction)
print('Result: ')
print(final_prediction)
print('Bettar than 0.891 & 0.905.')