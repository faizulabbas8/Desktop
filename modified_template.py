"""
This program finds the best value of K in KMeans algorithm using Silhouette Coefficient for 'housing.csv' dataset. The range of K values to analyze is provided as a command line parameter.
Syntax: python assignment.py <number> <number>

For example, to search best K between 3 and 6 the command line input should be:
python assignment.py 3 6
"""

# importing the libraries

"""  DO NOT MODIFY  """
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
"""  DO NOT MODIFY  """

def find_best_kmeans(data, min_k, max_k):

    """  write from here  """
    
    # Load the dataset
    df = pd.read_csv(data)
    
    best_k = None
    best_score = -1

    # Loop through the range of K values
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
        labels = kmeans.fit_predict(df)
        silhouette_avg = metrics.silhouette_score(df, labels)
        
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_k = k

    # Return the best K
    return best_k

    """        End        """


"""  DO NOT MODIFY  """
if __name__ == '__main__':

    """
    ALERT: * * * No changes are allowed in this section  * * *
    """
 
    if len(sys.argv) == 3:
        try:
            min_k = int(sys.argv[1])
            max_k = int(sys.argv[2])
            print(find_best_kmeans('housing.csv', min_k, max_k))
        except ValueError:
            print("Please provide valid integers for the range.")
    else:
        print("Please provide exactly two integers for the range.")
