'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
np.random.seed(4)
class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        result = []
        point_set = self.l2_distance(test_point)

        

        while (k>0):
            
            closest_point_index = np.nanargmin(point_set) # index of closest point
            result.append(closest_point_index)

            # remove the already selected point from candidates
            point_set = np.where(point_set==min(point_set),np.nan,point_set)
            k = k - 1
        label_counts = {0.:0,1.:0,2.:0,3.:0,4.:0,5.:0,6.:0,7.:0,8.:0,9.:0}
        
        for points in result:
            label_counts[self.train_labels[points]] += 1

        highest = max(label_counts.values())
        digits = [k for k, v in label_counts.items() if v == highest]
        if len(digits) == 1:
            return digits[0]
        else:
            prob_dist = []
            for i in range(len(digits)):
                prob_dist.append(1/len(digits))
            result = int(np.random.choice(digits,1,p = prob_dist))
            return result

def cross_validation(knn, k_range=np.arange(1,16)):
    result = []
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        
        avg_accuracy = 0
        kf = KFold(n_splits=10,shuffle = True)
        for train_index, test_index in kf.split(knn.train_data):
            train_data,test_data = knn.train_data[train_index],knn.train_data[test_index]
            train_labels,test_labels = knn.train_labels[train_index],knn.train_labels[test_index]

            cros_knn = KNearestNeighbor(train_data, train_labels)
            accuracy = classification_accuracy(cros_knn,k,test_data,test_labels)           
            avg_accuracy = avg_accuracy + accuracy
        avg_accuracy = avg_accuracy/10
        result.append(avg_accuracy)
        print("For k value of {}, the avg accuracy across the fold is {}".format(k,avg_accuracy))
    plt.plot(result)
    plt.show()
    plt.close()
    index = np.argmax(result) + 1
    return index

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    correct = 0
    for i in range(eval_data.shape[0]):
        digit = knn.query_knn(eval_data[i],k)
        if (digit == eval_labels[i]):
            correct += 1
        
    return correct/eval_data.shape[0]

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    print("The train accuracy for K=1 is {}".format(classification_accuracy(knn,1,train_data,train_labels)))
    print("The Test accuracy for K=1 is {}".format(classification_accuracy(knn,1,test_data,test_labels)))
    print("The train accuracy for K=15 is {}".format(classification_accuracy(knn,15,train_data,train_labels)))
    print("The Test accuracy for K=15 is {}".format(classification_accuracy(knn,15,test_data,test_labels)))
    optimal_k = cross_validation(knn, k_range=np.arange(1,16))
    print("The optimal K is {}".format(optimal_k))
    print(classification_accuracy(knn,optimal_k,test_data,test_labels))
    


