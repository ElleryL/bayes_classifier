'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)


def regularization(train_data,train_labels):

    noise_train_data = train_data
    print(noise_train_data.shape)
    noise_train_labels = train_labels
    noise_data_off = np.zeros(64).reshape(1,64)
    noise_data_on = np.ones(64).reshape(1,64)
    for i in range(10):

        noise_train_data = np.concatenate((noise_train_data,noise_data_off),axis=0)
        noise_train_data = np.concatenate((noise_train_data,noise_data_on),axis=0)
        noise_train_labels = np.concatenate((noise_train_labels,np.array([i])),axis=0)
        noise_train_labels = np.concatenate((noise_train_labels,np.array([i])),axis=0)

    return noise_train_data,noise_train_labels

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    for i in  range(0,10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        for j in range(0,train_data.shape[1]):
            eta[i][j] = (np.sum(i_digits[:,j])+2-1)/(i_digits.shape[0]+2+2-2)
    
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    eta_plt = []
    for i in range(10):
        img_i = class_images[i]
        img_i = img_i.reshape(8,8)
        eta_plt.append(img_i)
    all_concat = np.concatenate(eta_plt, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for i in range(10):
        for j in range(64):
            generated_data[i][j] = np.random.binomial(1,eta[i][j])
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    #here we assume that we take bin_digits as (700,64), a set of a class of data
    log_eta_T = np.log(eta.T)
    log_one_minus_eta_T = np.log(1-eta.T)
    result = np.dot(bin_digits,log_eta_T) + np.dot(1-bin_digits,log_one_minus_eta_T)
    
    return result

def prob_x(bin_digits, eta):
    p_x = np.zeros(bin_digits.shape[0])
    p_x_given_y = np.exp(generative_likelihood(bin_digits, eta))
    for i in range(10):
        p_x += p_x_given_y[:,i] * (1/10)
    return p_x
def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    #here we assume that we take bin_digits as (700,64), a set of a class of data
    result = np.zeros((bin_digits.shape[0],eta.shape[0]))
    log_p_x_given_y = generative_likelihood(bin_digits, eta)
    class_prob = np.zeros((bin_digits.shape[0],10)) + (1/10)
    log_joint = log_p_x_given_y + np.log(class_prob)
    p_x = prob_x(bin_digits, eta)
    log_p_x = np.log(p_x)
    for i in range(10):
        result[:,i] = log_joint[:,i] - log_p_x   
    return result

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits,eta)
    result = 0
    for i in range(bin_digits.shape[0]):
        result += cond_likelihood[i][int(labels[i])]
    result = result/bin_digits.shape[0] 
        
    return result
    
    return result

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    post_class = np.argmax(cond_likelihood,axis = 1)

    # or random sample from this dist
##    post_class2 = np.zeros(bin_digits.shape[0])
##    for i in range(bin_digits.shape[0]):
##        post_class2[i] = np.random.choice(np.arange(10), 1,p=np.exp(cond_likelihood[i]))[0]
    return post_class

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)

    generate_new_data(eta)

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')



    
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)        
    
    
    eta = compute_parameters(train_data, train_labels)
    plot_images(eta)
    generate_new_data(eta)
    i_digits = data.get_digits_by_label(train_data, train_labels, 0)
    p_x_given_y = generative_likelihood(i_digits, eta)
    p_y_given_x = conditional_likelihood(i_digits, eta)

    print("For the test accuracy")
    overall_correct = 0
    for i in range(10):
        i_digits = data.get_digits_by_label(test_data, test_labels, i)
        x = generative_likelihood(i_digits,eta)
        z = conditional_likelihood(i_digits, eta)
        result = classify_data(i_digits, eta)
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
        #avg = avg_conditional_likelihood(i_digits, i, means, covariances)
        #print("The avg log prob is {}".format(avg))
        x = generative_likelihood(i_digits,eta)
        z = conditional_likelihood(i_digits, eta)
        result = classify_data(i_digits, eta)
        correct = 0
        for j in result:
            if j == i:
                correct += 1
        overall_correct += correct
        print("The train accuracy for {}th class is {}".format(i, correct/i_digits.shape[0]))
    print("Over all train accuracy rate is {}".format(overall_correct/train_data.shape[0]))


    print("Average Log Likelihood for train: {}".format(avg_conditional_likelihood(train_data,train_labels,eta)))
    print("Average Log Likelihood for test: {}".format(avg_conditional_likelihood(test_data,test_labels,eta)))
