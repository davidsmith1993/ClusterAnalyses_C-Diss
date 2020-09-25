# ClusterAnalyses_C-Diss

## Analysis Goals

1. Cluster analysis

- Cluster rrs, erq_reap, erq_sup, and ders
- Create a column with cluster values for each participant id
'''

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix 

dfmodel = pd.read_csv('C:/Users/Pablo/Documents/Data Challenges/Colin Diss/inlab-subset.csv', index_col='id')
dfmodel = dfmodel.dropna()



#Create the models and the labels with 2 clusters
scaler = StandardScaler()
scaler.fit(dfmodel)
dfmodel_scaled = scaler.transform(dfmodel)

#choose the nmber of clusters
clusters = 2
#create the model and fit it
model = KMeans(n_clusters = clusters)
model.fit(dfmodel_scaled)
#create the labels
labels = model.predict(dfmodel_scaled)
#Check model inertia
print(model.inertia_)



#This part plots the inertias for the different number of clusters (but the model created has 2 clusters)
ks = range(1, 30)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)
    
    # Fit model to samples
    model.fit(dfmodel_scaled)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
#plt.xlabel('number of clusters, k')
#plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


#add lebels back on
dfmodel_labeled = dfmodel
dfmodel_labeled['cluster_label'] = labels
dfmodel_labeled_scaled = dfmodel_scaled

dfmodel_labeled_scaled = pd.DataFrame(dfmodel_labeled_scaled)
dfmodel_labeled_scaled['cluster_label'] = labels

'''
Analysis goal 2 -
- KNN of how the power state levels map onto the clusters
- Alternatively, can try linear discrimination analysis (LDA)
'''



#Point to the power bsc
power = pd.read_csv('C:/Users/Pablo/Documents/Data Challenges/Colin Diss/power-081520.csv', index_col='id')
power['UTC time'] = pd.to_datetime(power['UTC time'])
#drop the event column for now
power = power.drop(columns = ['event'])

#create a dataframe with the power and created clusters
dfall = dfmodel_labeled.merge(power, how = 'outer', left_index = True, right_index = True)

#save as a csv
#dfall.to_csv(r'C:/Users/Pablo/Documents/Data Challenges/Colin Diss/colllindissdata.csv') 


#you can delete this, but the g is the the info for different clusters
g = dfall.groupby('cluster_label').mean()
g2 = dfmodel_labeled_scaled.groupby('cluster_label').mean()




#Create training test data
#drop the NAs
dfall = dfall.dropna()
#set the test and training data
y = dfall['cluster_label']
X = dfall[['level', 'timestamp']]
X['timestamp'] = X['timestamp'] / 1000


# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42, stratify=y)



#This fits based on 1-10 neighbors and prints out the scores for each
for x in (range(1, 10)):
# Create a k-NN classifier with 7 neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = x)

# Fit the classifier to the training data
    knn.fit(X_train, y_train)

# Print the accuracy
    accuracy = (knn.score(X_test, y_test))
    print('n_neighbor = ' + str(x) +' accuracy: ' + str(accuracy))
    
    y_pred = knn.predict(X_test)
    c = confusion_matrix(y_test,y_pred)
    
    sensitivity = c[0][0] / (c[0][0] + c[1][0])
    specificity = c[1][1] / (c[1][1] + c[0][1])
    print('n_neighbor = ' + str(x) +' specificity:' + str(specificity))
    print('n_neighbor = ' + str(x) +' sensitivity:' + str(sensitivity))
    





