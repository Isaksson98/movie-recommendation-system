import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
#from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from scipy import spatial
import time


#each row in dara: u,m,r: 

train_data = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/filis220.training", header=None)
test_data = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/filis220.test", header=None)

task2_test_data = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/task---2.test", header=None)
task2_train_data = pd.read_csv("C:/Users/Filip/Skola/TSKS33/lab2/data/task---2.training", header=None)

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

A = create_matrix_A(task2_train_data)
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

    print('RMSE (test):')
    #print(RMSE_test)
    rms = math.sqrt(mean_squared_error(test_ratings, r_prediction_arr))
    print(rms)
    #RMSE_train = np.sqrt(diff_train**2)
    r_pred = np.rint(r_prediction_arr).astype(int)[0]
    
    r = test_ratings-np.transpose(r_pred)
    rating_plot = np.abs(r) 

    #print('H: ', rating_plot)
    #print('H[0]: ', rating_plot[0])

    #fig = plt.figure(figsize =(10, 7)) 
    #plt.hist(rating_plot, bins=[1,2,3,4,5])  # `density=False` would make counts
    #plt.ylabel('Number')
    #plt.xlabel('Error');
    #plt.title('Absolute errors')
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
    print('RMSE (train):')
    #print(RMSE_train)
    print(rms2)
    return [r_prediction_arr, target] #test, train

[baseline_prediction_test, baseline_prediction_train] = task1(task2_train_data, task2_test_data)

###TASK 2###
def task2(data_set, baseline_prediction, d, r_tilde_matrix):
    user_ids = np.array(data_set[ data_set.columns[0]])
    movie_ids = np.array(data_set[ data_set.columns[1]])
    ratings = np.array(data_set[ data_set.columns[2]])

    L = 100

    #Calculates the correction term
    correction = []
    correction = np.asarray( correction_term(d, L, r_tilde_matrix, user_ids, movie_ids) )


    final_prediction = np.add(correction, np.transpose(baseline_prediction))

    rms = math.sqrt(mean_squared_error(ratings, np.transpose( final_prediction )))
    
    return rms


def cosSimilarity(movie1_id, movie2_id, M):
   
    x = 0
    temp1=0
    temp2=0
    temp3=0
    res = 0

    u_min = 10
   
    ids1 = np.where( (M[:,movie1_id-1] == 0) ) #Index of all users that movies 1 that user didn't rate
    ids2 = np.where( (M[:,movie2_id-1] == 0) ) #index of all movies 2 that user didn't rate

    ids_tot = np.concatenate((ids1, ids2), axis=1) #Add them together

    # Remove all ratings that that isn't users that rated both films. Left with the overlap
    M_arr1 = np.delete(M[:,movie1_id-1], (ids_tot), axis=0) 
    M_arr2 = np.delete(M[:,movie2_id-1], (ids_tot), axis=0)
    
    if len(M_arr1) < u_min:
        x = 0
    else:
        #x = 1 - spatial.distance.cosine(M_arr1, M_arr2)
        for j in range(len(M_arr1)):
            temp1 += M_arr1[j]*M_arr2[j]
            temp2 += M_arr1[j]**2
            temp3 += M_arr2[j]**2
        x = temp1/(math.sqrt(temp2)*math.sqrt(temp3))

    return x

def correction_term(cos_sim, L, r_tilde, users, movies):
    
    num1=0
    num2=0
    correction = []
    for i in range(len(movies)):
        L_indecies = np.argsort(cos_sim[ movies[i]-1 ,:]) #Sort movies j for each movie i and get index
        a2 = L_indecies[-L:] #Indices for the largest L cosine movies

        for l in a2:
            num1 += cos_sim[ movies[i]-1 ][l] * r_tilde[ users[i]-1 ][l]
            num2 += abs(cos_sim[ movies[i]-1 ][l])

        
        #If L is too large and u_min is too large. one of the largest cos-sim
        # will include 0 ==> avoid division by zero
        if(num2==0):
            #print('zero')
            correction.append( 0 )
        else:
            correction.append( (num1/num2) )

        num1=0
        num2=0

    return correction


def create_rating_matrix(user_id, movie_id, ratings):

    M = np.zeros(shape=(max(user_id),max(movie_id)))
    for i in range(len(ratings)):
        M[user_id[i]-1,movie_id[i]-1]=ratings[i]
    
    return M

def calculate_d(data_set, baseline_prediction):
    user_ids = np.array(data_set[ data_set.columns[0]])
    movie_ids = np.array(data_set[ data_set.columns[1]])
    ratings = np.array(data_set[ data_set.columns[2]])

    r_tilde = np.subtract(ratings, np.transpose(baseline_prediction))
    r_tilde_matrix = create_rating_matrix(user_ids, movie_ids, r_tilde[0])

    cos_sim = []

    #Creates the cos-similarity matrix
    for m1 in range(1,max(movie_ids)+1):
        for m2 in range(1,max(movie_ids)+1):
            cos_sim.append( cosSimilarity(m1, m2, r_tilde_matrix))
       
    cos_sim = np.asarray(cos_sim)
    d = cos_sim.reshape(max(movie_ids),max(movie_ids))
    return [d, r_tilde_matrix]

[d, r_tilde_matrix] = calculate_d(task2_train_data, baseline_prediction_train)

final_prediction_rms_test = task2(task2_test_data, baseline_prediction_test, d, r_tilde_matrix)
print('RMSE (test) - improved: ')
print(final_prediction_rms_test)

final_prediction_rms_train = task2(task2_train_data, baseline_prediction_train, d, r_tilde_matrix)

print('RMSE (train) - improved: ')
print(final_prediction_rms_train)
