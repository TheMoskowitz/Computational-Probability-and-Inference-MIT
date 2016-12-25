import sys
import os.path
import numpy as np

import util

from collections import Counter

USAGE = "%s <test data folder> <spam folder> <ham folder>"

number_of_spam_emails = 0
number_of_ham_emails = 0
# This is a global variable that will represent either the 
# number of spam or ham emails depending on need.
number_of_emails = 0


def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    ### TODO: Comment out the following line and write your code here

    #raise NotImplementedError

    word_dict = Counter()

    for file in file_list:
        words = set(util.get_words_in_file(file))
        for item in words:
            word_dict[item] += 1

    return word_dict




def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    ### TODO: Comment out the following line and write your code here
    #raise NotImplementedError

    global number_of_emails
    number_of_emails = len(file_list) 
    word_dict = myDefaultDict(get_counts(file_list))

    for word in word_dict:
        word_dict[word] = np.log(word_dict[word] / number_of_emails)

    return word_dict

    # global number_of_emails
    # number_of_emails = len(file_list)
    # word_dict = Counter()
    # word_dict = get_counts(file_list)

    # for word in word_dict:
    #     word_dict[word] = np.log(word_dict[word] / number_of_emails)

    # return word_dict


class myDefaultDict(dict):

    def __getitem__(self, item):
        global number_of_emails
        if item in self:
            num = dict.__getitem__(self, item)
            return num
        else:
            return np.log(1 / (number_of_emails))
            

def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    ### TODO: Comment out the following line and write your code here
    #raise NotImplementedError

    # This will hold the Spam and Ham log probabilities, in that order
    log_probabilities_by_category = [None] * 2

    global number_of_emails
    global number_of_ham_emails
    global number_of_spam_emails

    number_of_spam_emails = len(file_lists_by_category[0])
    number_of_ham_emails = len(file_lists_by_category[1])

    total_emails = len(file_lists_by_category[0]) + len(file_lists_by_category[1])

    log_prior_by_category = [None] * 2

    # Spam - s
    log_prior_by_category[0] = np.log(len(file_lists_by_category[0]) / (total_emails))

    # Ham - s
    log_prior_by_category[1] = np.log((len(file_lists_by_category[1])) / (total_emails))

    # Spam
    number_of_emails = number_of_spam_emails
    log_probabilities_by_category[0] = get_log_probabilities(file_lists_by_category[0])
    number_of_spam_emails = number_of_emails

    # Ham
    number_of_emails = number_of_ham_emails
    log_probabilities_by_category[1] = get_log_probabilities(file_lists_by_category[1])
    number_of_ham_emails = number_of_emails

    return log_probabilities_by_category, log_prior_by_category


def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names.
    """
    ### TODO: Comment out the following line and write your code here
    #return 'spam'

    email_dict = set(util.get_words_in_file(email_filename))

    spam_prob = 0
    ham_prob = 0
    spam_comp = 0
    ham_comp = 0


    # # Spam
    # number_of_emails = number_of_spam_emails
    # for word in log_probabilities_by_category[0]:
    #     if word in email_dict:
    #         spam_prob += log_probabilities_by_category[0][word]
    #     else:
    #         num = np.exp(log_probabilities_by_category[0][word])
    #         spam_comp += np.log(1 - num)


    # # Ham
    # number_of_emails = number_of_ham_emails
    # for word in log_probabilities_by_category[1]:
    #     if word in email_dict:
    #         ham_prob += log_probabilities_by_category[1][word]
    #     else:
    #         num = np.exp(log_probabilities_by_category[1][word])
    #         ham_comp += np.log(1 - num)

    all_words = set(log_probabilities_by_category[0].keys())
    all_words.update(set(log_probabilities_by_category[1].keys()))
    all_words = set(all_words)

    number_of_emails = len(all_words)
    for word in all_words:
        if word in email_dict:
            spam_prob += log_probabilities_by_category[0][word]
            ham_prob += log_probabilities_by_category[1][word]
        else:

            spam_num = np.exp(log_probabilities_by_category[0][word])
            spam_comp += np.log(1-spam_num)
            ham_num = np.exp(log_probabilities_by_category[1][word])
            ham_comp += np.log(1-ham_num)

    log_spam_prob = log_prior_by_category[0] + spam_prob
    log_ham_prob = log_prior_by_category[1] + ham_prob

    if (log_spam_prob) - (log_ham_prob) > 0:
        return 'spam'
    else:
        return 'ham'

def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels

def main():
    ### Read arguments
    if len(sys.argv) != 4:
        print (USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

if __name__ == '__main__':
    main()
