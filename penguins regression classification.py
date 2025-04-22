#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:47:21 2024

@author: anon
"""
#%% Cell 1, importing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Establishing a colour scheme
colours = ["blue", "orange", "green"]

#Importing
path_link = "/Users/theorogers/Desktop/Assessments/AI/penguins.csv"
penguins = pd.read_csv(path_link)
penguins.head()

#%% Cell 2, NA checking and removal

#Looking for NA values
na_pengs = penguins[penguins.isna().any(axis=1)]
na_pengs.head()
print(f"number of N/As: {len(na_pengs)}")

#These penguins have no gender this cannot be computed by mean and there is sufficient remaining data, consequentially they are removed
penguins = penguins.dropna()

#%% Cell 3, feature changing

#Feature creation
penguins['is_female'] = (penguins['sex'] == 'female').astype(int)
penguins = penguins.drop('sex',axis=1)

#Creating a list to iterate through
breeds = penguins['species'].unique()
    
#Creating a list to iterate through
continuous_column_names = penguins.columns[3:7]

#%% Cell 4, Histograms

#Creating a grid of histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

#Iterate through columns of interest
for i, continuous_col in enumerate(continuous_column_names):
    title_string = continuous_col.replace("_", " ") #Remove underscores
    ax = axes[i]
    for breed in breeds:
        breed_df = penguins[penguins['species'] == breed] #Subsetting dfs
        ax.hist(breed_df[continuous_col], alpha=0.5, label=breed,)
        plt.rcParams['font.size'] =18
        
    ax.set_title(f'Spread of {title_string} by breed')
    ax.set_ylabel('Frequency',fontsize = 18)
    ax.set_xlabel(title_string,fontsize = 20)
    ax.legend()
    
plt.tight_layout()
plt.show()



#Female count
for breed in breeds:
    breed_df = penguins[penguins['species'] == breed]
    female_count = breed_df['is_female'].sum()
    percent_female = 100*female_count/len(breed_df)
    print(f'{female_count} out of {len(breed_df)} {breed} penguins ({round(percent_female,2) }%) are female.')


#%% Cell 5, Correlogram

correlation_matrix = penguins.iloc[:, 3:7].corr()

# Create a colour-coded correlogram using seaborn heatmap
plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.rcParams['font.size'] =18
plt.title('Correlogram of Continuous Measures', fontsize = 25)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)

plt.show()

#%% Cell 6, Train Test Split

#Creating a train val test split (70,15,15) by breed
penguins_train = []
penguins_test = []
penguins_val = []
penguins_test_val = []
for breed in breeds:

    globals()[f"peng_{breed}"] = penguins[penguins['species'] == breed]
    
    globals()[f"train_{breed}"], globals()[f"test_val_{breed}"] = train_test_split(globals()[f"peng_{breed}"], test_size=0.15, random_state=1)
    globals()[f"test_{breed}"], globals()[f"val_{breed}"] = train_test_split(globals()[f"test_val_{breed}"], test_size=0.5, random_state=1) #(split the val/test set in 2 equal sizes)
    
    penguins_test_val.append(globals()[f"test_val_{breed}"])
    penguins_train.append(globals()[f"train_{breed}"])
    penguins_test.append(globals()[f"test_{breed}"])
    penguins_val.append(globals()[f"val_{breed}"])
    
penguins_train = pd.concat(penguins_train)
penguins_test = pd.concat(penguins_test)
penguins_val = pd.concat(penguins_val)
penguins_test_val = pd.concat(penguins_test_val)

#%% Cell 7, Regression

#Deleting columns that are not included, to create datasets that work for models
def delete_cols(dataset):
    dataset_LM = dataset.copy()
    dataset_LM = dataset_LM.drop('year', axis=1)
    dataset_LM = dataset_LM.drop('island', axis=1)
    dataset_LM = dataset_LM.drop('rowid', axis=1)
    return dataset_LM

penguins_train_LM = delete_cols(penguins_train)
penguins_test_val_LM = delete_cols(penguins_test_val)
penguins_test_LM = delete_cols(penguins_test_val)
penguins_val_LM = delete_cols(penguins_test_val)



#Defining a function with various inputs including booleans, which determine outputs
def linear_model(penguins_train, penguins_val, lasso, show,alpha):
    
    #If show is true then a scatter plot is created
    if show:
        plt.figure(figsize=(10, 6))
    
    #An empty list of model coefficients to display
    model_coefs = []
    
    #Mean Residual Mean Squared Error, across breeds, metric for fine tuning \alpha
    rmses = []
    
    #Iterate over breeds
    for i, breed in enumerate(breeds):
        
        #Subset the original df
        breed_df = penguins_train[penguins_train['species'] == breed].copy()
        breed_df = breed_df.drop('species',axis=1)
        
        #Select X and y
        y = breed_df['flipper_length_mm']
        X = breed_df.drop('flipper_length_mm', axis=1)
        
        #Model definition, whether or not lasso is used
        if lasso == True:
            model = Lasso(alpha=alpha[i])
        else:
            model = LinearRegression()
        model.fit(X, y)
        
        #Creating a list of model coefficient information
        model_coefs.extend([(breed, coef_name, coef_value, colours[i]) for coef_name, coef_value in zip(X.columns, model.coef_)])
        
        #Plot the scatter plot
        if show:
            
            #Scatter plot
            plt.scatter(X.iloc[:, 0], y, label=breed, alpha=0.7)
            
            #Sorting X in order of the variable that is to be displayed on the x axis
            sorted_indices = X.iloc[:, 0].argsort()
            X_sorted = X.iloc[sorted_indices]
        
            #Predicted values
            y_values = model.predict(X_sorted)
        
            #Plotting Regression line
            plt.plot(X_sorted.iloc[:, 0], y_values, color=colours[i], linewidth=2)
            
        #Calculating RMSE on validation data
        breed_val_df = penguins_val[penguins_val['species'] == breed].copy()
        breed_val_df = breed_val_df.drop('species',axis=1)
        
        y_val = breed_val_df['flipper_length_mm']
        X_val = breed_val_df.drop('flipper_length_mm', axis=1)
        
        y_val_hat = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_val_hat))
        
        #Displays the RMSE
        if show:
            print(f'RMSE for {breed}: {round(rmse,2)}')
        rmses.append(rmse)
        
    #Scatter plot continued:
    if show:
        plt.xlabel("Bill Length (mm)")
        plt.ylabel('Flipper Length (mm)')
        plt.title('Scatter Plot with Regression Lines')
        plt.legend(fontsize='medium', markerscale=0.75, loc='upper left', title_fontsize='medium')
        plt.show()
        
    #Plot a bar chart of the coefficients, sorted by name
        coefs_df = pd.DataFrame(model_coefs, columns=['Breed', 'Coefficient', 'Value', 'Colour'])

        plt.figure(figsize=(12, 6))
        unique_coefs = coefs_df['Coefficient'].unique()
        bar_width = 0.3
     
        # Group by coefficient name and plot
        for idx, coef in enumerate(unique_coefs):
            subset = coefs_df[coefs_df['Coefficient'] == coef]
            for j, (index, row) in enumerate(subset.iterrows()):
                plt.bar(idx + j*bar_width, row['Value'], color=row['Colour'], width=bar_width, label=row['Breed'] if idx == 0 else "")
     
        plt.xticks([i + bar_width for i in range(len(unique_coefs))], unique_coefs, rotation=30, ha='right', fontsize = 20)
        plt.xlabel('Measure', fontsize = 22)
        plt.ylabel('Coefficient Value', fontsize = 22)
        plt.title('Model Coefficients by Breed', fontsize = 25)
        plt.legend(title="Breed")
        plt.tight_layout()
        plt.show()
    else:
        return(rmses)
    

#Candidate alpha values
alpha_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  

results = {}

for alpha in alpha_range:
    alphas = (alpha, alpha, alpha)
    rmses = linear_model(penguins_train_LM, penguins_val_LM, lasso=True, show=False, alpha=alphas)
    results[alphas] = rmses
    
#Optimised alphas
alpha = [0.5,0.01,0.5]

#Finalised model:
linear_model(penguins_train_LM,penguins_test_LM, lasso = True, show = True, alpha = alpha)
    
#%% Cell 8, Baseline

#Baseline prediction model
def baseline(train_df):
    breed_counts = {}
    for breed in train_df["species"].unique():
        count = len(train_df[train_df["species"] == breed])
        breed_counts[breed] = count
    return(max(breed_counts, key=breed_counts.get))
    
penguins_test_val["baseline_pred"] = baseline(penguins_train)

#Confusion matrix plot
def plot_conf_mat(df_test,pred_col,figure_title,ret_accuracy):
    conf_mat = confusion_matrix(penguins_test_val["species"], pred_col)
    accuracy = 100*np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
    if not ret_accuracy:
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='g', 
                    xticklabels=penguins_test_val["species"].unique(),
                    yticklabels=penguins_test_val["species"].unique())
        plt.xlabel('Predicted', fontsize = 18)
        plt.ylabel('Actual', fontsize = 18)
        plt.title(f'Confusion Matrix for a {figure_title}')
        plt.show()
        print(f"Accuracy = {round(accuracy,2)}%")
    else:
        return(accuracy)
    
    
#Confusion matrix on baseline prediction model
plot_conf_mat(penguins_test_val,penguins_test_val["baseline_pred"],"baseline prediction",ret_accuracy=False)

#%% Cell 9, K-NN

from sklearn.neighbors import KNeighborsClassifier

#Defining the KNN function
def K_nn(penguins_train, penguins_test, n_neighbours,ret_plot_conf,ret_accuracy):
        
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(penguins_train.drop(columns='species'))
    X_test = scaler.transform(penguins_test.drop(columns='species'))
    y_train = penguins_train['species']
    
    k_nn = KNeighborsClassifier(n_neighbors=n_neighbours)
    
    k_nn.fit(X_train,y_train)
    
    k_nn_pred = k_nn.predict(X_test)
    if ret_accuracy:
        accuracy = plot_conf_mat(penguins_test, k_nn_pred, "K Nearest Neigbours", ret_accuracy=ret_accuracy)
        return(accuracy)
    else:
        if ret_plot_conf:
            plot_conf_mat(penguins_test, k_nn_pred, "K Nearest Neigbours", ret_accuracy=ret_accuracy)
        else:
            return(k_nn_pred)

#Candidate number of neighbours to use
for i in range(1,8):
    print(f"Using {i} neighbours gives accuracy of: {K_nn(penguins_train_LM, penguins_val_LM, i,False,True)}")
    
#Chosen model, test performance
K_nn(penguins_train_LM, penguins_test_LM, 3,True,False)
#%% Cell 10, R-forest
from sklearn.ensemble import RandomForestClassifier

#Candidate hyper parameters:
hyper_params = {
    'n_estimators': range(1,11),  # Number of decision trees in the random forest
    'max_depth': [None, 10, 20],  # MThe maximum depth of the tree
    'min_samples_split': [2, 4, 6],  # The minimum number of data points of a class requited to justify a to split
    'min_samples_leaf': [1, 2, 3, 4]  # Minimum number of samples required to be at a leaf node
}

#Random forest function
def r_forest(penguins_train, penguins_test, hyper_params, ret_plot_conf, ret_accuracy, figure_title):
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(penguins_train.drop(columns='species'))
    X_test = scaler.transform(penguins_test.drop(columns='species'))
    y_train = penguins_train['species']
    
    #Optimising on accuracy
    best_accuracy = 0
    best_params = {}
    
    #Iterating through every choice of hyper-parameters
    for n_estimators in hyper_params['n_estimators']:
        for max_depth in hyper_params['max_depth']:
            for min_samples_split in hyper_params['min_samples_split']:
                for min_samples_leaf in hyper_params['min_samples_leaf']:
                    rf = RandomForestClassifier(n_estimators=n_estimators, 
                                                max_depth=max_depth, 
                                                min_samples_split=min_samples_split, 
                                                min_samples_leaf=min_samples_leaf, 
                                                random_state=1)
                    
                    rf.fit(X_train, y_train)
                    rf_pred = rf.predict(X_test)
                    accuracy = plot_conf_mat(penguins_test, rf_pred, f"Random Forest {figure_title}", ret_accuracy=True)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
    
    if ret_accuracy:
        return best_accuracy, best_params
    else:
        if ret_plot_conf:
            plot_conf_mat(penguins_test, rf_pred, f"Random Forest {figure_title}", ret_accuracy=False)
        else:
            return rf_pred
#Choosing best parameters
r_forest(penguins_train_LM, penguins_val_LM, hyper_params, ret_plot_conf = False, ret_accuracy = True,figure_title = "")       

best_hyper_params = {
    'n_estimators': [10], 
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

#Test perfomance on optimised parameters
r_forest(penguins_train_LM, penguins_test_LM, best_hyper_params, ret_plot_conf = True, ret_accuracy = False,figure_title ="")  
