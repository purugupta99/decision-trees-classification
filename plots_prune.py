#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
from collections import Counter
from pprint import pprint
import copy
from random import randint
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


# In[35]:

df = pd.read_csv('test.csv', keep_default_na=False, header=None)
df.columns = ['job_type', 'degree', 'marriage_status', 'job', 'family', 'ethnicity', 'gender', 'country', 'salary']
cnt = Counter(x for x in df['salary'])
baseline_accuracy = cnt[' <=50K']/10000 

# print(baseline_accuracy)

percent = [40, 50, 60, 70, 80]

accuracy_train = []
accuracy_test = []
node_num = []

for p in percent:
# Read Train Data
    train_percent = p
    validate_percent = 20

    train_filename = 'train.csv'
    df = pd.read_csv(train_filename, keep_default_na=False, header=None)

    df_train = df[ : int(train_percent * len(df)/100)]
    df_validate = df[-int(validate_percent * len(df)/100) : ]

    # print(len(df_train))
    # print(len(df_validate))


    test_filename = 'test.csv'
    df_test = pd.read_csv(test_filename, keep_default_na=False, header=None)

    df_train.columns = ['job_type', 'degree', 'marriage_status', 'job', 'family', 'ethnicity', 'gender', 'country', 'salary']
    df_validate.columns = ['job_type', 'degree', 'marriage_status', 'job', 'family', 'ethnicity', 'gender', 'country', 'salary']
    df_test.columns = ['job_type', 'degree', 'marriage_status', 'job', 'family', 'ethnicity', 'gender', 'country', 'salary']

    node_number = 0


    # In[3]:


    # print(df)


    # In[4]:


    #Function to calculate the entropy of probability distribution
    # Sum (-p*log2*p)

    def get_entropy(probability):
        entropies = -probability * np.log2(probability) 
        return np.sum(entropies)

    # Entropy of given attr list with respect to the target attr
    def entropy_of_attribute(attr_list):
        cnt = Counter(x for x in attr_list)
        num_vals = len(attr_list)
        
        probability = []
        for key, val in cnt.items():
            probability.append(val/num_vals)
        
        probability = np.array(probability)
        
        return get_entropy(probability)

    # total_entropy = entropy_of_attribute(df_train['salary'])
    # print(total_entropy)


    # In[5]:


    # Calculate the information gain based on the split_attr with respect to the target_attr

    def information_gain(df, split_attribute_name, target_attribute_name):
        df_split = df.groupby(split_attribute_name)
        
    #     for attr_val, data_subset in df_split:
    #         print(attr_val, data_subset)
    #     print(df_split)
        
        observations = len(df.index)
    #     print(observations)
        
        df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_attribute, lambda x: len(x)/observations] })[target_attribute_name]
        df_agg_ent.columns = ['Entropy', 'Probability']
    #     print(df_agg_ent)
        
        new_entropy = np.sum(df_agg_ent['Entropy'] * df_agg_ent['Probability'])
        old_entropy = entropy_of_attribute(df[target_attribute_name])
        return old_entropy - new_entropy
        
        
    # degree = information_gain(df, 'degree', 'salary')
    # print(degree)


    # In[6]:


    def id3(df, target_attribute_name, attribute_names, default_class=None):
        
        global node_number 
        cnt = Counter(x for x in df[target_attribute_name])
        
        if len(cnt) == 1:
            return list(cnt.keys())[0]  
        elif df.empty or (not attribute_names):
            return default_class
        else:
            default_class = max(cnt.keys())
            info_gain = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
            max_gain_idx = np.argmax(info_gain)
            best_attr = attribute_names[max_gain_idx]
            
            tree = {best_attr:{}} # Initiate the tree with best attribute as a node 
            less_than_50 = df['salary'].value_counts()[0]
            more_than_50 = df['salary'].value_counts()[1]
            
            if more_than_50 > less_than_50:
                best_class = ' >50K'
            elif less_than_50 > more_than_50:
                best_class = ' <=50K'
            else:
                best_class = None
                
            node_number = node_number + 1
            
            tree[best_attr]['number'] = node_number
            tree[best_attr]['best_class'] = best_class
            
            remaining_attribute_names = [i for i in attribute_names if i != best_attr]
            
            for attr_val, data_subset in df.groupby(best_attr):
                subtree = id3(data_subset, target_attribute_name, remaining_attribute_names, default_class)
                tree[best_attr][attr_val] = subtree
            return tree


    # In[7]:


    def classify(instance, tree, default=None):
        attribute = next(iter(tree))
        if instance[attribute] in tree[attribute].keys(): # Value of the attributs in  set of Tree keys  
            result = tree[attribute][instance[attribute]]
    #         print("Instance Attribute:",instance[attribute],"TreeKeys :",tree[attribute].keys())
            if isinstance(result, dict): # this is a tree, delve deeper
                return classify(instance, result)
            else:
                return result # this is a label
        else:
            return default


    # In[8]:


    def preorder (temptree, number):
        if isinstance(temptree, dict):
            attribute = list(temptree.keys())[0]
    #         print(attribute)
            if temptree[attribute]['number'] == number:
                
                for key, val in temptree[attribute].items():
                    if isinstance(val, dict):
                        temp_tree = val
                        if isinstance(temp_tree, dict):
                            temp_attribute = list(temp_tree.keys())[0]
                            temptree[attribute][key] = temp_tree[temp_attribute]['best_class']

            else:
                children = []
                for key, val in temptree[attribute].items():
                    children.append(val)
                    
                for c in children:
                    preorder(c, number)
                
        return temptree


    # In[9]:


    # leaf_num = []
    def count_number_of_non_leaf_nodes(tree):
        if isinstance(tree, dict):
    #         print('non - leaf ', tree)
            attribute = list(tree.keys())[0]
    #         print(attribute)
            
            children = []
    #         is_leaf = True
            for key, val in tree[attribute].items():
    #             if isinstance(val, dict):
    #                 is_leaf = False
                children.append(val)
    #         if is_leaf == True:
    #             leaf_num.append(tree[attribute]['number'])
            
            count = []
            for c in children:
                count.append(count_number_of_non_leaf_nodes(c))
            return (1 + np.sum(count))
        else:
            return 0


    # In[10]:


    # count_number_of_non_leaf_nodes(tree)
    # print(leaf_num)


    # In[11]:


    def post_prune(df_validate, L, K, tree):
        best_tree = tree
        for i in range(1, L+1) :
            temp_tree = copy.deepcopy(best_tree)
            M = randint(1, K);
            for j in range(1, M+1):
                n = count_number_of_non_leaf_nodes(temp_tree)
                if n> 0:
                    P = randint(1,n)
                else:
                    P = 0
                temp_tree = preorder(temp_tree, P)
            df_validate['accuracyBeforePruning'] = df_validate.apply(classify, axis=1, args=(best_tree,'1') ) 
            accuracyBeforePruning = str( np.sum(df_validate['salary']==df_validate['accuracyBeforePruning'] ) / (len(df_validate.index)) )
            df_validate['accuracy_after_pruning'] = df_validate.apply(classify, axis=1, args=(temp_tree,'1') ) 
            accuracy_after_pruning = str( np.sum(df_validate['salary']==df_validate['accuracy_after_pruning'] ) / (len(df_validate.index)) )
            
            # print(accuracy_after_pruning, accuracyBeforePruning)
            
            if accuracy_after_pruning > accuracyBeforePruning:
                best_tree = temp_tree
        return best_tree


    # In[12]:


    attribute_names = list(df_train.columns)
    attribute_names.remove('salary') # Remove target attr
    # print(attribute_names)

    tree = id3(df_train, 'salary', attribute_names)
    pruned_tree = post_prune(df_validate, 100, 5, tree)
    
    df_train['predicted'] = df_train.apply(classify, axis=1, args=(pruned_tree,'No'))
    df_test['predicted'] = df_test.apply(classify, axis=1, args=(pruned_tree,'No'))
    
    acc_train = sum(df_train['salary']==df_train['predicted'] ) / (len(df_train.index))
    acc_test = sum(df_test['salary']==df_test['predicted'] ) / (len(df_test.index))
    
    node_num.append(count_number_of_non_leaf_nodes(pruned_tree))
    accuracy_train.append(acc_train)    
    accuracy_test.append(acc_test)


# In[ ]:


print(accuracy_test)

# [0.6607, 0.6521, 0.6444, 0.6359]


# In[ ]:


plt.plot(percent, accuracy_train, label='Train Accuracy')
plt.plot(percent, accuracy_test, label='Test Accuracy')
plt.plot(percent, [baseline_accuracy]*5, label='Baseline Accuracy')

plt.title('Accuracy vs Train Set Percentage for Pruned Tree')

plt.legend(loc='upper right')
plt.show()


# In[ ]:


plt.plot(percent, node_num, label='Number of Nodes')

plt.title('Number of Nodes vs Train Set Percentage for Pruned tree')

plt.legend(loc='upper right')
plt.show()

