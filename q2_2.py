'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import scipy.stats

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(0,10):
      i_digits = data.get_digits_by_label(train_data, train_labels, i)
      i_mean = (sum(i_digits[:,])/i_digits.shape[0])
      means[i] = i_mean
    
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    means = compute_mean_mles(train_data, train_labels)
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    # sample variance fomular
    for i in range(0,10):
      i_digits = data.get_digits_by_label(train_data, train_labels, i)
      i_mean = means[i]

      i_cov = np.dot((i_digits-i_mean).T,(i_digits-i_mean))/(i_digits.shape[0])
      stable = np.identity(i_digits.shape[1])*0.01
      i_cov = i_cov + stable
      covariances[i] = i_cov      
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    cov = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        cov_diag = np.log(cov_diag.reshape(8,8))
        cov.append(cov_diag)
    all_concat = np.concatenate(cov,1)
    plt.imshow(all_concat,cmap = 'gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    result = np.zeros((digits.shape[0], 10)) 
    
    d = means.shape[1]    
    for i in range(0,10):
      const = -d/2*np.log(2*np.pi) - 1/2*(np.log(np.linalg.det(covariances[i])))     
      i_mean = means[i]
      inverse_cov = np.linalg.inv(covariances[i])
      for j in range(digits.shape[0]):
          non_const = -1/2*np.dot(np.dot((digits[j,:] - i_mean).T,inverse_cov),(digits[j,:] - i_mean))
          log_p_x_given_y = const + non_const
          result[j][i] = log_p_x_given_y

    #gaussians = np.array([scipy.stats.multivariate_normal(means[i], covariances[i]) for i in range(10)])
    #gaussians = np.log(np.array([gaussians[i].cdf(digits) for i in range(10)]).T)
    return result


def prob_x(digits, means, covariances):
    p_x = np.zeros(digits.shape[0])
    p_x_given_y = np.exp(generative_likelihood(digits, means, covariances))
    for i in range(10):
        p_x += p_x_given_y[:,i] * (1/10)
    return p_x

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    
    class_prob = np.zeros((digits.shape[0],10)) + (1/10)
    log_p_x_given_y = generative_likelihood(digits, means, covariances)

    log_joint = log_p_x_given_y + np.log(class_prob)
    p_x = prob_x(digits, means, covariances)
    log_p_x = np.log(p_x)
    result = np.zeros((digits.shape[0],10))
    for i in range(10):
        result[:,i] = log_joint[:,i] - log_p_x
    return result




def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    
    # Compute as described above and return

        
    result = 0
    for i in range(digits.shape[0]):
        result += cond_likelihood[i][int(labels[i])]
    result = result/digits.shape[0] 
        
    return result

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    # result is (N,10), where for each data point, there is ten probability
    post_class = np.argmax(cond_likelihood,axis = 1)
    
    
    return post_class




if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    plot_cov_diagonal(covariances)


    # test_accuracy
    print("For the test accuracy")
    overall_correct = 0
    for i in range(10):
        i_digits = data.get_digits_by_label(test_data, test_labels, i)
        z = conditional_likelihood(i_digits, means, covariances)
       
        result = classify_data(i_digits, means, covariances)
        correct = 0
        for j in result:
            if j == i:
                correct += 1
        overall_correct += correct
        print("The test accuracy for {}th class is {}".format(i, correct/i_digits.shape[0]))
    print("Over all test accuracy rate is {}".format(overall_correct/test_data.shape[0]))

    print("For the train accuracy")
    overall_correct = 0
    for i in range(10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)

        x = generative_likelihood(i_digits,means,covariances)
        z = conditional_likelihood(i_digits, means, covariances)

        result = classify_data(i_digits, means, covariances)
        correct = 0
        for j in result:
            if j == i:
                correct += 1
        overall_correct += correct
        print("The train accuracy for {}th class is {}".format(i, correct/i_digits.shape[0]))
    print("Over all train accuracy rate is {}".format(overall_correct/train_data.shape[0]))


    print("The Over ALl Average Log Likelihood for train: {}".format(avg_conditional_likelihood(train_data,train_labels,means,covariances)))


    print("The Over ALl Average Log Likelihood for test: {}".format(avg_conditional_likelihood(test_data,test_labels,means,covariances)))
    
##################

