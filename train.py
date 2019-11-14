# Importing Librarires
import numpy as np
from numpy import array
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from statistics import mean 
from datavisualisation import tsne_plot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np 
from sklearn import cluster
from sklearn import metrics
import pandas as pd
import seaborn as sns  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

all_accuracies=[]
classifiers=[]
all_val_acc=[]

choice = 1  # 0,1,2

# Logistic regression from scratch

class LogisticRegression_fromscratch:
    def __init__(self, lr, num_iter, threshold = 0.5):
        self.lr = lr
        self.num_iter = num_iter
        self.threshold = threshold


    def __add_intercept(self,X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)


    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def train(self,X,y):
        X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
           z = np.dot(X, self.theta)
           h = self._sigmoid(z)
           gradient = np.dot(X.T, (h - y)) / y.size
           self.theta -= self.lr * gradient

           #z = np.dot(X, self.theta)
           #h = self._sigmoid(z)
           loss = self.__loss(h, y)

           if( i % 500 == 0):
               print("Training Loss :")
               print(loss)
               
        filehandler = open("logistic_regression_model.pickle",'wb')
        pickle.dump(self.theta,filehandler)
        filehandler.close()

        return True

    def predict(self, X):
        file = open("logistic_regression_model.pickle",'rb') #has theta values
        self.theta = pickle.load(file)
        X = self.__add_intercept(X)
        prob = self._sigmoid(np.dot(X, self.theta))
        return prob >= self.threshold
    def evaluate(self, test_x, test_y):
        y_predicted = self.predict(test_x)
        correct = 0
        for i,y in enumerate(test_y):
            if y == 0:
                y = False
            else:
                y = True
            if y == y_predicted[i]:
                correct = correct + 1
        total = y_predicted.size

        return (correct/total)*100


# Plot dataset distribution
if choice == 0 : data = pd.read_csv('datasets/sms_spam1.csv', encoding = 'latin1')
elif choice == 1 : data = pd.read_csv('datasets/sms_spam2.csv', encoding = 'latin1')
elif choice == 2 : data = pd.read_csv('datasets/sms_spam3.csv', encoding = 'latin1')
else : print("Wrong choice")


data.type.value_counts()
sns.countplot(x = "type", data = data)
data.loc[:, 'type'].value_counts()
plt.title('Distribution of Spam and Not-spam')
plt.savefig("dataset_distribution_simple.png", dpi=1200,  bbox_inches='tight') 
plt.close()


# This function will read the data and appends data to a list
def read(filename):
    all_messages = []
    all_categories = []
    df = pd.read_csv(filename)
    messsages = (df.iloc[:,1:2])
    categories = (df.iloc[:,0:1])
    messsages_to_list = (messsages.values.tolist())
    categories_to_list = (categories.values.tolist())

    for message in messsages_to_list:
        all_messages.append(message[0])
    for category in categories_to_list:
        if category[0] == "spam":
            all_categories.append(1)
        else:
            all_categories.append(0)


    return (all_messages, all_categories)


# This function will read data and convert string to vectors
def create_unique_dict(text_arr):
    dictt = {}
    count = -1
    x = np.array(text_arr)
    text1 = (np.unique(x))

    for item in text1:
        data = (item.lower().strip().split(" "))
        for item1 in data:
            if item1 not in dictt:
                count = count +1
                dictt[item1] = count
    return dictt

def vectrorize(text_arr,dictt):
    vectors  = []
    for sentence in text_arr:
        vector = [0] * len(dictt)
        words = (sentence.lower().strip().split(" "))
        for word in words:
            index = dictt[word]
            vector[index] += 1
        vectors.append(vector)
    #print(array(vectors))
    return vectors

#start
# Loading total text
if choice == 0 : 
    total_x, total_y = read("data/sms_spam1.csv")
elif choice == 1 : 
    total_x, total_y = read("data/sms_spam2.csv")
elif choice == 2 : 
    total_x, total_y = read("data/sms_spam3.csv")
else : print("Wrong choice")

dictt = create_unique_dict(total_x)

x, x_test, y, y_test = train_test_split(total_x, total_y, test_size=0.2)
#print X_train.shape, y_train.shape
#print X_test.shape, y_test.shape


# Creating dictionary from total text to pickle
filehandler = open("dictionary.pickle",'wb')
pickle.dump(dictt,filehandler)
filehandler.close()

#====================================================================================
x = np.array(vectrorize(x, dictt))
y = np.array(y)
#===================================================================================

# Changeable variables
lr=0.0001
num_iter=100
K = 10
logr = LogisticRegression_fromscratch(lr,num_iter, 0.5)
#===================================================================================

#10 fold cross validation for logistic regression from scratch

import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
kf.get_n_splits(x)

print(kf)  

KFold(n_splits=K, random_state=None, shuffle=False)
a = 0
for train_index, test_index in kf.split(x):
    logr.train(x[train_index],y[train_index])
    print("Validation Accuracy :")
    b = logr.evaluate(x[test_index], y[test_index])
    a = a + b
    print(b)
    

#reading test data and vectorizing it


test_x = np.array(vectrorize(x_test, dictt))
test_y = np.array(y_test)

print("\nK-fold (10-fold) Validation Accuracy: ")
print(a/K)
all_val_acc.append(a/K)
print("\nTesting Accuracy: ")
acc_log=logr.evaluate(test_x,test_y)
print(acc_log)
all_accuracies.append(acc_log)
classifiers.append("Logistic regression from scratch")
from statistics import mean

clfs = {
    
    'gnb': GaussianNB(),
    'svm1': SVC(kernel='linear'),
    'svm2': SVC(kernel='rbf'),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'lr': LogisticRegression()
}

# vectorising text data
xtestvec = np.array(vectrorize(x_test, dictt))
ytestvec = np.array(y_test)

# inbuilt models

for clf_name in clfs:
    print("\n")
    print(clf_name)
    clf = clfs[clf_name]
    scores = cross_val_score(clf, x, y , cv=10)
    clf.fit(x,y)
    y_pred = clf.predict(xtestvec)
    print("\nK-Fold (10-Fold) Validation Score: ")
    print(mean(scores))
    all_val_acc.append(mean(scores)*100)
    acc=accuracy_score(ytestvec, y_pred)*100
    print("Accuracy score = ",'%.2f'%(acc))
    all_accuracies.append(acc)
    classifiers.append(clf_name)
    print(confusion_matrix(ytestvec, y_pred, labels=[0,1]))

a1 = np.arange(7)
plt.plot(a1, all_accuracies, c="blue", label="Testing Accuracies") 
plt.plot(a1, all_val_acc, c="red", label="Validation Accuracies") 
plt.title("Test Accuracy Vs Valdation Accuracy Vs Model")
plt.xlabel("Various Models") 
plt.ylabel("Accuracies") 
plt.xticks(a1, classifiers, rotation ="vertical") 
plt.ylim(0,100)
plt.legend() 
plt.grid(color="gray") 
plt.savefig("All_accuracies_graph.png", dpi=1200,  bbox_inches='tight') 
plt.close() 


sentences=[]
for i in total_x:
    sentences.append(i.split(' '))
print(np.shape(sentences))
print(sentences[0])  
  
model = Word2Vec(sentences, size=100, window=20, min_count=1, workers=4)
print("going to call tsne plotting...")
tsne_plot(model)


#####################################################KMEANS CLUSTER ##########################################
  
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw
  
  
X=[]
for sentence in sentences:
    X.append(sent_vectorizer(sentence, model))   
   
  

  

NUM_CLUSTERS=2
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)
  
  
  
for index, sentence in enumerate(sentences):    
    print (str(assigned_clusters[index]) + ":" + str(sentence))
 
     
     
     
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
  
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
  
print ("Cluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)


#The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters 
silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

print ("Silhouette_score: ")
print (silhouette_score)
 
 

 
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
 
Y=model.fit_transform(X)
 
 
plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290,alpha=.5)
 
 
for j in range(len(sentences)):    
   plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
   #print ("%s %s" % (assigned_clusters[j],  sentences[j]))
 
 
plt.savefig("clustering.png", dpi=1200,  bbox_inches='tight') 


#==================================================
