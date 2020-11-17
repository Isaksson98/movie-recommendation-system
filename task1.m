filename='../data/verification.training';
veri = csvread(filename);

users = veri(:,1);
movies = veri(:,2);
ratings = veri(:,3);

nr_ratings=length(ratings)
u_max=max(users)
m_max=max(movies)

A = sparse([1:nr_ratings 1:nr_ratings]', [users; movies+u_max], ones(2*nr_ratings,1),nr_ratings,u_max+m_max);


