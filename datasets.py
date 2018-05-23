import pandas as pd
from sklearn.model_selection import train_test_split

# LOAD THE DATASETS
df1 = pd.read_csv('data/data_Sha_64.txt', header=None) # shape descriptor
df2 = pd.read_csv('data/data_Tex_64.txt', header=None) # texture histogram
df3 = pd.read_csv('data/data_Mar_64.txt', header=None) # fine scale margin

# PREP THE DATASETS
def prep_data(df):
    # map species names to discrete values
    mapping = {v:k for k,v in enumerate(df[0].unique())}
    df.replace({0: mapping}, inplace=True)

    # shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    return df

df1 = prep_data(df1)
df2 = prep_data(df2)
df3 = prep_data(df3)

# SPLIT THE DATASETS INTO TRAINING AND TESTING SETS
X1 = df1.drop([0], axis=1)
y1 = df1[0]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, shuffle=False)

X2 = df2.drop([0], axis=1)
y2 = df2[0]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, shuffle=False)

X3 = df3.drop([0], axis=1)
y3 = df3[0]
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, shuffle=False)