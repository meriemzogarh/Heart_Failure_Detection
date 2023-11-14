#!/usr/bin/env python
# coding: utf-8

# <h3>Importing Packages</h3>

# In[1]:


# Handling Data
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Machine Learning
from sklearn.preprocessing import StandardScaler # Scaling
from sklearn.preprocessing import OrdinalEncoder # Convert Categorical Data
from sklearn.model_selection import train_test_split # Train Test Split

# Metrics
from sklearn.metrics import accuracy_score # Scoring Models Accuracy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Models
from sklearn.neighbors import KNeighborsClassifier as KNN # K-Nearest Neighbors
from sklearn.svm import SVC                               # Support Vector Classifier
from sklearn.ensemble import RandomForestClassifier as RF # Random Forest Classifier
from sklearn.tree import DecisionTreeClassifier as DT     # Decision Tree Classifier


# <h3>Importing Data</h3>

# In[ ]:


path = r'heart.csv'
df = pd.read_csv(path)

df.head()


# <h3>Data Shape </h3>

# In[ ]:


print(f'{df.shape[0]} Rows, {df.shape[1]} Columns')


# <h3> Data types information </h3>

# In[ ]:


df.dtypes


# In[ ]:


import matplotlib.pyplot as plt

l = list(df['HeartDisease'].value_counts())
circle = [l[1], l[0]]

categories = ['No Heart Disease', 'Heart Disease']
colors = ['#9e2a2b', '#023e8a']

# Create subplots with adjusted aspect ratio
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1.5]})

# Pie chart
wedges, texts, autotexts = ax1.pie(circle, labels=None, autopct='%1.1f%%', startangle=90, explode=(0.1, 0),
                                   colors=colors, wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
ax1.set_title('Percentage of patients with or without Heart Disease')

# Create legend for the pie chart with larger font size
ax1.legend(wedges, categories, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2,
           prop={'size': 12})  # Adjust the font size as desired

# Bar plot with total numbers
ax2.bar(categories, circle, color=colors)
ax2.set_xlabel('Categories')
ax2.set_ylabel('Number of People')
ax2.set_title('Number of patients with or without Heart Disease')

# Add numbers on top of each bar
for i in range(len(categories)):
    ax2.text(i, circle[i], str(circle[i]), ha='center', va='bottom')

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.5)

# Show the combined plot
plt.show()


# <h3>Create a copy of the dataframe</h3>

# In[ ]:


df1 = df.copy()


# In[ ]:


df1.describe()


# In[ ]:


df1.isna().sum()


# In[ ]:


df1


# In[ ]:


from sklearn.preprocessing import LabelEncoder

column_name = 'ST_Slope'  # Replace 'your_column_name' with the actual name of the column

# Create a new LabelEncoder instance
label_encoder = LabelEncoder()

# Fit the LabelEncoder to the column and transform the values
encoded_labels = label_encoder.fit_transform(df1[column_name])

# Retrieve the original categorical values
original_values = label_encoder.inverse_transform(encoded_labels)

# Print the mapping between original values and encoded labels
for original, encoded in zip(original_values, encoded_labels):
    print(f'{original} --> {encoded}')


# In[ ]:


df1


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in df1.select_dtypes(include=['object']).columns:
    df1[i] = label_encoder.fit_transform(df1[i])


# In[ ]:


df1


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# Set custom color palette
custom_palette = ['#9e2a2b', '#023e8a']

# Create count plot
sns.countplot(data=df1, x='HeartDisease', hue='Sex', palette=custom_palette)

# Set labels and title
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.title('Number of Male and Female Patients by Heart Disease Category')

# Customize legend labels
legend_labels = ['Female', 'Male']
plt.legend(title='Sex', labels=legend_labels)

# Add numbers on top of the bars
for p in plt.gca().patches:
    height = p.get_height()
    plt.gca().annotate(str(height), (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')

# Show the plot
plt.show()


# In[ ]:


df1.dtypes


# In[ ]:


df1.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16,9))
sns.heatmap(df1.corr(),annot=True,cmap="coolwarm")


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate the correlation matrix
corr_matrix = df1.corr()

# Calculate the number of rows and columns
N, M = corr_matrix.shape

# Extract the labels for the x-axis and y-axis
ylabels = corr_matrix.index.tolist()
xlabels = corr_matrix.columns.tolist()

# Create a colormap that matches the "coolwarm" colormap
cmap = sns.color_palette("inferno", as_cmap=True)

# Generate the heatmap without annotations
fig, ax = plt.subplots(figsize=(16, 9))
sns.heatmap(corr_matrix, annot=False, cmap=cmap, linewidths=0.5, linecolor='lightgray', cbar=True, ax=ax)

# Create circle markers with sizes and colors based on the correlation values
circle_radius = 0.25
circle_colors = ['white', 'gray']  # Customize the circle colors as desired

for i in range(N):
    for j in range(M):
        correlation_value = corr_matrix.iloc[i, j]
        circle = plt.Circle((j + 0.5, i + 0.5), radius=circle_radius, facecolor=circle_colors[int(correlation_value < 0)], edgecolor='gray')
        ax.add_patch(circle)
        text_color = 'black'
            
        ax.text(j + 0.5, i + 0.5, f'{correlation_value:.2f}', ha='center', va='center', color=text_color)

# Set the tick labels
ax.set_xticks(np.arange(M) + 0.5)
ax.set_yticks(np.arange(N) + 0.5)
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)

# Remove the grid lines
ax.grid(False)

# Show the plot
plt.show()


# In[ ]:


df1["ChestPainType"].dtypes


# In[ ]:


import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd

df_filtered = df1[(df1["Age"] > 10) & (df1["Age"] < 99)]

# Create subplots with 1 row and 2 columns
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("With Disease", "Without Disease"))

# Add histogram trace for with disease class
fig.add_trace(go.Histogram(
    x=df_filtered[df_filtered["HeartDisease"] == 1]["Age"],
    marker=dict(color="blue"),
    opacity=0.7,
    nbinsx=10,
), row=1, col=1)

# Add histogram trace for without disease class
fig.add_trace(go.Histogram(
    x=df_filtered[df_filtered["HeartDisease"] == 0]["Age"],
    marker=dict(color="red"),
    opacity=0.7,
    nbinsx=10,
), row=1, col=2)

# Update layout
fig.update_layout(
    title={
        'text': "Age distribution by Class",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Age",
    yaxis_title="Count",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#000000"
    ),
    showlegend=False  # Hide legend
)

# Show the subplot
fig.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Set up the figure with an increased figsize
fig, axs = plt.subplots(ncols=2, nrows=6, figsize=(12, 40))
axs = axs.flatten()

palette=['#9e2a2b', '#023e8a']
colors = sns.color_palette(palette)

# Loop through each variable and create a violin plot
for i, column in enumerate(df1.columns[:-1]):
    ax = axs[i]
    sns.violinplot(x='HeartDisease', y=column, data=df1, ax=ax, palette=colors)
    
    # Create custom legend handles
    legend_elements = [
        Patch(facecolor=colors[0], label='Healthy'),
        Patch(facecolor=colors[1], label='Sick')
    ]
    
    # Add legend at the top of the plot
    legend = ax.legend(handles=legend_elements, title='Legend', loc='upper center', bbox_to_anchor=(0.5, 1))
    
    
    # Adjust subplot layout
    plt.subplots_adjust(hspace=0.2)

# Show the plot
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set up the figure size
plt.figure(figsize=(20, 40))

# Specify the number of plots per row
plots_per_row = 2

# Calculate the number of rows required for the plots
num_rows = int(np.ceil(len(df1.columns[:-1]) / plots_per_row))
# Specify the minimum count threshold
min_count_threshold = 10 

# Iterate over the selected variables
for i, column in enumerate(df1.columns[:-1]):
    # Calculate the counts for each value in the column
    value_counts = df1[column].value_counts()

    # Filter out values that do not meet the minimum count threshold
    significant_values = value_counts[value_counts >= min_count_threshold].index

    # Subset the data based on the significant values
    selected_data = df1[df1[column].isin(significant_values)]

    # Check if there are enough significant values to plot
    if len(significant_values) > 0:
        # Create a subplot for the current variable
        ax = plt.subplot(num_rows, plots_per_row, i+1)

        # Create the count plot
        sns.countplot(x=column, hue='HeartDisease', data=selected_data, palette='bright', ax=ax)

        # Add labels to the bars
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height):
                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontsize=10)

        # Set labels and title
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {column} by Heart Disease')

        ax.legend(labels=['Healthy Patient', 'Sick Patient'])

        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # Set font size
        ax.tick_params(axis='both', labelsize=10)

         # Adjust subplot layout
        plt.subplots_adjust(hspace=0.2)
# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# <h2>Data Preparation</h2>

# In[ ]:


df1.isna().sum()


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming df1 is your DataFrame
non_missing_values_count = df1.notna().sum()

# Plotting the non-missing values count
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
non_missing_values_count.plot(kind='bar')
plt.title('Non-Missing Values Count')
plt.xlabel('Columns')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# <h3>Duplicated Data</h3>

# In[ ]:


f"Number of Duplicated Rows {df1.duplicated().sum()}"


# In[ ]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have already imported the necessary libraries and have the DataFrame `df1` available

# Calculate the count of duplicated rows
duplicated_count = df1.duplicated().sum()
unique_count = len(df1) - duplicated_count

# Create a bar plot using Seaborn
ax = sns.barplot(x=['Duplicated Rows', 'Unique Rows'], y=[duplicated_count, unique_count])

# Set the plot labels and title
plt.xlabel('Row Type')
plt.ylabel('Count')
plt.title('Duplicated Rows in DataFrame')

# Add the count values on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

# Show the plot
plt.show()


# <h3>Correlation between features and heart disease</h3>

# In[ ]:


df_corr = df1.corr()
df_corr[["HeartDisease"]].sort_values(by=["HeartDisease"])


# <h3>Correlation HeatMap</h3>

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select correlations of "HeartDisease" column
heart_disease_corr = df1.corr()["HeartDisease"].drop("HeartDisease")

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heart_disease_corr.to_frame(), annot=True, cmap="inferno", vmin=-1, vmax=1)
plt.title("Correlation Heatmap - HeartDisease")
plt.show()


# <h3>Separate Data</h3>

# In[ ]:


df1.dtypes


# In[ ]:


num_cols = df1.select_dtypes(include=['int64', 'float64']).columns.tolist()[:-1] #Removing [HeartDisease]
cate_cols = df1.select_dtypes(include=['int32']).columns.tolist()
target = ['HeartDisease']
print("Numerical Columns: ", num_cols)
print("Categorical Columns: ", cate_cols)
print("Target Column: ", target)


# <h2>Feature Scaling</h2>

# In[ ]:


df1.dtypes


# In[ ]:


scaler = StandardScaler() #scaling in range [-1, 1]
standard_df = scaler.fit_transform(df1[num_cols])
standard_df = pd.DataFrame(standard_df, columns = num_cols)
standard_df


# <h3>Merge the new columns to the categorical columns</h3>

# In[ ]:


df1 = standard_df.join(df1[cate_cols+target])
df1


# In[ ]:


df1.describe()


# <h2>Building Models</h2>

# Split the data into 70% training and 30% testing (to build and evaluate Models)

# In[ ]:


X = df1.drop(target, axis=1)
y = df1[target]
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create a LogisticRegression classifier object
logreg = LogisticRegression()

# Fit the logistic regression model to the training data
logreg.fit(x_train, y_train)

# Make predictions on the test data
y_pred_logreg = logreg.predict(x_test)

# Calculate the accuracy score of the logistic regression classifier
acc_logreg = accuracy_score(y_test, y_pred_logreg)

print('Accuracy score of Logistic Regression Classifier is', acc_logreg)


# In[ ]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_logreg)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Oranges', fmt='d')

# Add labels, title, and ticks to the plot
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Logistic Regression Confusion Matrix')
plt.xticks(ticks=[0, 1, 2])  # Replace with the appropriate class labels
plt.yticks(ticks=[0, 1, 2])  # Replace with the appropriate class labels

# Display the plot
plt.show()


# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create an SVC classifier object
svc = SVC()

# Fit the SVC model to the training data
svc.fit(x_train, y_train)

# Make predictions on the test data
y_pred_svc = svc.predict(x_test)

# Calculate the accuracy score of the SVC classifier
acc_svc = accuracy_score(y_test, y_pred_svc)

print('Accuracy score of SVC Classifier is', acc_svc)


# In[ ]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_svc)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Purples', fmt='d')

# Add labels, title, and ticks to the plot
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('SVC Confusion Matrix')
plt.xticks(ticks=[0, 1, 2])  # Replace with the appropriate class labels
plt.yticks(ticks=[0, 1, 2])  # Replace with the appropriate class labels

# Display the plot
plt.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create a KNN classifier object
knn = KNeighborsClassifier()

# Fit the KNN model to the training data
knn.fit(x_train, y_train)

# Make predictions on the test data
y_pred_knn = knn.predict(x_test)

# Calculate the accuracy score of the KNN classifier
acc_knn = accuracy_score(y_test, y_pred_knn)

print('Accuracy score of KNN Classifier is', acc_knn)


# In[ ]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_knn)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Oranges', fmt='d')

# Add labels, title, and ticks to the plot
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('KNN Confusion Matrix')
plt.xticks(ticks=[0, 1, 2])  # Replace with the appropriate class labels
plt.yticks(ticks=[0, 1, 2])  # Replace with the appropriate class labels

# Display the plot
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create a Random Forest classifier object
rf = RandomForestClassifier()

# Fit the Random Forest model to the training data
rf.fit(x_train, y_train)

# Make predictions on the test data
y_pred_rf = rf.predict(x_test)

# Calculate the accuracy score of the Random Forest classifier
acc_rf = accuracy_score(y_test, y_pred_rf)

print('Accuracy score of Random Forest Classifier is', acc_rf)


# In[ ]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Greens', fmt='d')

# Add labels, title, and ticks to the plot
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Random Forest Classifier Confusion Matrix')
plt.xticks(ticks=[0, 1, 2])  # Replace with the appropriate class labels
plt.yticks(ticks=[0, 1, 2])  # Replace with the appropriate class labels

# Display the plot
plt.show()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create a Decision Tree classifier object with max_depth=4
dt = DecisionTreeClassifier(max_depth=4)

# Fit the Decision Tree model to the training data
dt.fit(x_train, y_train)

# Make predictions on the test data
y_pred_dt = dt.predict(x_test)

# Calculate the accuracy score of the Decision Tree classifier
acc_dt = accuracy_score(y_test, y_pred_dt)

print('Accuracy score of Decision Tree Classifier is', acc_dt)


# In[ ]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_dt)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

# Add labels, title, and ticks to the plot
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Decision Tree Confusion Matrix')
plt.xticks(ticks=[0, 1, 2])  # Replace with the appropriate class labels
plt.yticks(ticks=[0, 1, 2])  # Replace with the appropriate class labels

# Display the plot
plt.show()


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create a Naive Bayes classifier object
nb = GaussianNB()

# Fit the Naive Bayes model to the training data
nb.fit(x_train, y_train)

# Make predictions on the test data
y_pred_nb = nb.predict(x_test)

# Calculate the accuracy score of the Naive Bayes classifier
acc_nb = accuracy_score(y_test, y_pred_nb)

print('Accuracy score of Naive Bayes Classifier is', acc_nb)


# In[ ]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_nb)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Reds', fmt='d')

# Add labels, title, and ticks to the plot
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Naive Bayes Confusion Matrix')
plt.xticks(ticks=[0, 1, 2])  # Replace with the appropriate class labels
plt.yticks(ticks=[0, 1, 2])  # Replace with the appropriate class labels

# Display the plot
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

models = ["Logistic Regression", "Support Vector Classifier", "K-Nearest Neighbors", "Random Forest Classifier", "Decision Tree Classifier", "Naive Bayes"]
scores = [acc_logreg, acc_svc, acc_knn, acc_rf, acc_dt, acc_nb]

colors = ["#184e77", "#1e6091", "#1a759f", "#168aad", "#34a0a4", "#52b69a"]

# Sort the models and scores based on scores in descending order
sorted_scores, sorted_models = zip(*sorted(zip(scores, models), reverse=True))

fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figsize for a larger plot
plt.subplots_adjust(left=0.1)  # Adjust the left border

ax.barh(sorted_models, sorted_scores, color=colors)
plt.xlabel("Accuracy", fontweight='bold')
plt.ylabel("classifiers", fontweight='bold')
plt.title("Comparison of Accuracy values for different classifiers", fontweight='bold')

for i, score in enumerate(sorted_scores):
    ax.text(score, i, f"{score:.2f}", ha='left', va='center')

plt.show()


# In[ ]:


# Import the required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd

# Define the set of classifiers
models = [LogisticRegression(),SVC(), KNeighborsClassifier(), RandomForestClassifier(),
           DecisionTreeClassifier(max_depth=4), GaussianNB()]

# Check the correctness of the list of classifiers
model_names = [type(model).__name__ for model in models]
print(model_names)

# Define a function to evaluate classifiers
def classifiers(models):
    columns = ['Score', 'PPV', 'TNR', 'Predictions']  # Added 'Predictions' to columns
    df_result = pd.DataFrame(columns=columns, index=[type(model).__name__ for model in models])
   
    for model in models:
        clf = model
        print('Initialized classifier {} with default parameters \n'.format(type(model).__name__))    
        clf.fit(x_train, y_train)
        #make a predicitions for entire data(X_test)
        predictions = clf.predict(x_test)
        # Use score method to get accuracy of model
        score = clf.score(x_test, y_test)
        print('Score of classifier {} is: {} \n'.format(type(model).__name__, score))
        #df_result['Score']['{}'.format(type(model).__name__)] = str(round(score * 100, 2)) + '%' 
        df_result['Score']['{}'.format(type(model).__name__)] = score
        df_result['Predictions']['{}'.format(type(model).__name__)] = predictions

        confusion_matr=confusion_matrix(y_test,predictions)
        TP = confusion_matr[0,0]
        FP = confusion_matr[0,1]
        FN = confusion_matr[1,0]
        TN = confusion_matr[1,1]
        df_result['PPV']['{}'.format(type(model).__name__)] = TP / (TP + FN) #positive predictive value
        df_result['TNR']['{}'.format(type(model).__name__)] = TN / (TN + FP)  #true negative rate
    return df_result


# In[ ]:


classifiers(models)


# In[ ]:



import matplotlib.pyplot as plt
import numpy as np


# Call the classifiers function with your data
result = classifiers(models)

# Extract the required data from the result DataFrame
scores = result['Score'].astype(float)
ppv = result['PPV'].astype(float)
tnr = result['TNR'].astype(float)

# Plotting the bar plot
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(models))
width = 0.20


bars = ax.bar(x, scores, width, label='Score', color='#ff9f1c')
bars1 = ax.bar(x + width, ppv, width, label='PPV', color='#2ec4b6')
bars2 = ax.bar(x + width * 2, tnr, width, label='TNR', color='#e71d36')

# Add the value labels on top of each bar
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(bars)        
autolabel(bars1)
autolabel(bars2)

ax.set_xlabel('Classifiers', fontweight='bold')
ax.set_ylabel('Values', fontweight='bold')
ax.set_xticks(x + width * 1.5)

# Set the x-tick labels in two lines
labels = ['\n'.join(label.split()) for label in result.index]
ax.set_xticklabels(labels, rotation=0)

# Move the legend to the top right without covering the last bar
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Add a title to the plot
ax.set_title('Comparison of the Accuracy, the PPV, and the TNR values for different classifiers', fontweight='bold')

plt.tight_layout()
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function to plot confusion matrix
def plot_confusion_matrix(ax, confusion_matrix, title, cmap, accuracy):
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'{title}\n\nAccuracy: {accuracy:.2%}', fontweight='bold')

# Evaluate classifiers and collect confusion matrices
confusion_matrices = []
model_names = []
model_accuracies = []
cmap_list = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'Greys']  # List of color palettes

for i, model in enumerate(models):
    clf = model
    print('Initialized classifier {} with default parameters \n'.format(type(model).__name__))
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    confusion_matr = confusion_matrix(y_test, predictions)
    accuracy = clf.score(x_test, y_test)
    confusion_matrices.append(confusion_matr)
    model_names.append(type(model).__name__)
    model_accuracies.append(accuracy)

# Create a figure and subplots
num_models = len(models)
num_cols = 3  # Number of columns in the subplot grid
num_rows = int(np.ceil(num_models / num_cols))  # Number of rows in the subplot grid
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

# Plot confusion matrix for each model with a different color palette and accuracy percentage
for i in range(num_models):
    ax = axes[i // num_cols, i % num_cols]
    cmap = cmap_list[i % len(cmap_list)]  # Select a color palette from the list
    plot_confusion_matrix(ax, confusion_matrices[i], model_names[i], cmap, model_accuracies[i])

    # Add model name and accuracy in bold on top of the confusion matrix
    #ax.text(0.5, 1.15, model_names[i], fontweight='bold', fontsize=12, ha='center', transform=ax.transAxes)
    #ax.text(0.5, 1.05, f'Accuracy: {model_accuracies[i]:.2%}', fontweight='bold', fontsize=12, ha='center', transform=ax.transAxes)

# Remove empty subplots if the number of models is not a multiple of num_cols
if num_models % num_cols != 0:
    if num_rows > 1:
        for j in range(num_models % num_cols, num_cols):
            fig.delaxes(axes[num_rows-1, j])
    else:
        for j in range(num_models % num_cols, num_cols):
            fig.delaxes(axes[j])

# Adjust the spacing between subplots and display the plot
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
### LogisticRegression
# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = KFold(n_splits=10)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


### KNeighborsClassifier
# define models and parameters
model = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
# define grid search
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = KFold(n_splits=3)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


### SVC
# define model and parameters
model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = KFold(n_splits=3)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


### RandomForestClassifier
# define models and parameters
model = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = KFold(n_splits=3)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

