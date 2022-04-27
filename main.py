import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(file_name: str):
    df = pd.read_csv(file_name)
    df_columns = df.columns.values.tolist()

    features = df_columns[0:14]
    label = df_columns[14:]

    X = df[features]
    y = df[label]

    y = pd.get_dummies(y)  # one-hot

    # Question 2 / 3
    print(df['Class'].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=42)

    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)


if __name__ == '__main__':
    prepare_data('data/synthetic.csv')
