import numpy as np
import pandas as pd
import os

def read_data():
    raw_data_path = os.path.join(os.path.pardir, 'data', 'raw')
    train_data_path = os.path.join(raw_data_path, 'train.csv')
    test_data_path = os.path.join(raw_data_path, 'test.csv')
    
    train_df =  pd.read_csv(train_data_path, index_col='PassengerId')
    test_df = pd.read_csv(test_data_path, index_col='PassengerId')
    test_df['Survived'] = -888
    df = pd.concat((train_df, test_df), axis=0)
    return df

def process_data(df):
    return(df
          .assign(Title = lambda x: x.Name.map(get_title))
          .pipe(fill_missing_values)
          .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']))
          .assign(Age_State = lambda x: np.where(x.Age >= 18, 'Adult', 'Child'))
          .assign(Family_Size = lambda x: x.Parch + x.SibSp + 1)
          .assign(IsMother = lambda x: np.where(((x.Sex == 'female')&(x.Age >= 18)&(x.Title != 'Miss')&(x.Parch > 0)), 1, 0))
          .assign(Cabin = lambda x: np.where(x.Cabin == 'T', np.nan, x.Cabin))
          .assign(Deck = lambda x: x.Cabin.map(get_deck))
          .assign(IsMale = lambda x: np.where(x.Sex == 'male', 1, 0))
          .pipe(pd.get_dummies, columns=['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'Age_State'])
          .drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis=1)
          .pipe(reorder_cols))

def get_title(name):
    title_group = {
        'mr': 'Mr',
        'mrs': 'Mrs',
        'miss':'Miss',
        'master':'Master',
        'don':'Sir',
        'rev': 'Sir',
        'dr':'Officer',
        'mme': 'Mrs',
        'ms':'Mrs',
        'major':'Officer',
        'lady':'Lady',
        'sir':'Sir',
        'mlle': 'Miss',
        'col':'Officer',
        'capt':'Officer',
        'the countess':'Lady',
        'jonkheer':'Sir',
        'dona':'Lady'
    }
    full_name = name.split(',')[1]
    title = full_name.split('.')[0]
    title = title.strip().lower()
    return title_group[title]

def get_deck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')

def fill_missing_values(df):
    df.Embarked.fillna('C', inplace=True)
    median_fare = df[(df.Pclass == 3) & (df.Embarked == 'S')]['Fare'].median()
    df.Fare.fillna(median_fare, inplace=True)
    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median, inplace=True)
    return df

def reorder_cols(df):
    cols = [col for col in df.columns if col != 'Survived']
    cols = ['Survived'] + cols
    df = df[cols]
    return df

def write_data(df):
    processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')
    train_path = os.path.join(processed_data_path, 'train.csv')
    test_path = os.path.join(processed_data_path, 'test.csv')
    df[df.Survived != -888].to_csv(train_path)
    cols = [col for col in df.columns if col != 'Survived']
    df[df.Survived == -888][cols].to_csv(test_path)
    
if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)
