import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth

base_dir = 'D:\\Colin\\Programming\\DataSets\\fifa19\\'


def run_all():
    df = get_df()
    # df = groom_df()
    description = df.describe()
    print(description.to_string())
    print()
    cluster_data(df)
    i = 0


def get_df():
    df = pd.read_csv(base_dir + 'groomed_data.csv', encoding='utf-8')
    return df


def groom_df():
    # Get the data frame
    df = pd.read_csv(base_dir + 'data.csv', encoding='utf-8')
    df = df.loc[df['Work Rate'].notnull()]
    df = df.loc[df['Position'].notnull()]
    df = df.loc[df['Club'].notnull()]

    df = df.fillna(0)

    # Drop the unnecessary columns
    df = df.drop(columns=['Unnamed: 0', 'Photo', 'Flag', 'Club Logo'])

    # Encode the nationality.
    nationality_label_encoder = LabelEncoder()
    nationality_label_encoder = nationality_label_encoder.fit(df['Nationality'])
    df['Nationality ID'] = nationality_label_encoder.transform(df['Nationality'])

    # Encode the club
    club_label_encoder = LabelEncoder()
    club_label_encoder = club_label_encoder.fit(df['Club'])
    df['Club ID'] = club_label_encoder.transform(df['Club'])

    # Fix value and wage
    def fix_value(row, col_name):
        value = row[col_name]
        value = str(value).replace('€', '')
        if 'M' in value:
            value = value.replace('M', '')
            value = float(value)
            value = value * 1e6
        elif 'K' in value:
            value = value.replace('K', '')
            value = float(value)
            value = value * 1e3
        return value

    df['Value'] = df.apply(lambda row: fix_value(row, 'Value'), axis=1)
    df['Wage'] = df.apply(lambda row: fix_value(row, 'Wage'), axis=1)
    df['Release Clause'] = df.apply(lambda row: fix_value(row, 'Release Clause'), axis=1)

    # Encode the preferred foot
    foot_map = {'Right': 1,
                'Left': 0}
    df['Preferred Foot'] = df['Preferred Foot'].map(foot_map)

    # Encode the work rates
    df['Attack Work Rate'] = df.apply(lambda row: row['Work Rate'].split('/ ')[0], axis=1)
    df['Defense Work Rate'] = df.apply(lambda row: row['Work Rate'].split('/ ')[1], axis=1)

    work_rate_map = {'High': 2,
                     'Medium': 1,
                     'Low': 0}
    df['Attack Work Rate'] = df['Attack Work Rate'].map(work_rate_map)
    df['Defense Work Rate'] = df['Defense Work Rate'].map(work_rate_map)

    # Fix Height and Weight
    df['Height'] = df.apply(lambda row: int(row['Height'].split("'")[0]) * 12 + int(row['Height'].split("'")[1]),
                            axis=1)
    df['Weight'] = df.apply(lambda row: row['Weight'].replace('lbs', ''), axis=1)
    df['Height'] = df['Height'].astype(np.int64)
    df['Weight'] = df['Weight'].astype(np.int64)

    # Encode the body type
    body_type_map = {'Messi': 'Lean',
                     'C. Ronaldo': 'Stocky',
                     'Neymar': 'Lean',
                     'Courtois': 'Stocky',
                     'PLAYER_BODY_TYPE_25': 'Lean',
                     'Shaqiri': 'Lean',
                     'Lean': 'Lean',
                     'Normal': 'Normal',
                     'Stocky': 'Stocky',
                     'Akinfenwa': 'Stocky'}
    df['Body Type'] = df['Body Type'].map(body_type_map)
    body_label_encoder = LabelEncoder()
    body_label_encoder = body_label_encoder.fit(df['Body Type'])
    df['Body Type ID'] = body_label_encoder.transform(df['Body Type'])

    # Encode the faces
    real_face_map = {'Yes': 1,
                     'No': 0}
    df['Real Face'] = df['Real Face'].map(real_face_map)

    # Encode the position
    position_label_encoder = LabelEncoder()
    position_label_encoder = position_label_encoder.fit(df['Position'])
    df['Position ID'] = position_label_encoder.transform(df['Position'])

    def fix_joined(row):
        import maya
        joined_str = row['Joined']
        if str(joined_str) == '0':
            joined_str = row['Contract Valid Until']
        joined = maya.when(joined_str)
        epoch = joined.epoch
        return epoch

    df['Joined'] = df.apply(lambda row: fix_joined(row), axis=1)

    def fix_valid_until(row):
        import maya
        joined_str = row['Contract Valid Until']
        joined = maya.when(joined_str)
        epoch = joined.epoch
        return epoch

    df['Contract Valid Until'] = df.apply(lambda row: fix_valid_until(row), axis=1)

    def fix_position_skill_rating(row, position):
        rating = row[position]
        if '+' in str(rating):
            base = rating.split('+')[0]
            comp = rating.split('+')[1]
            rating = int(base) + int(comp)
        else:
            rating = int(rating)
        return rating

    df['LS'] = df.apply(lambda row: fix_position_skill_rating(row, 'LS'), axis=1)
    df['ST'] = df.apply(lambda row: fix_position_skill_rating(row, 'ST'), axis=1)
    df['RS'] = df.apply(lambda row: fix_position_skill_rating(row, 'RS'), axis=1)

    df['LW'] = df.apply(lambda row: fix_position_skill_rating(row, 'LW'), axis=1)
    df['LF'] = df.apply(lambda row: fix_position_skill_rating(row, 'LF'), axis=1)
    df['CF'] = df.apply(lambda row: fix_position_skill_rating(row, 'CF'), axis=1)
    df['RF'] = df.apply(lambda row: fix_position_skill_rating(row, 'RF'), axis=1)
    df['RW'] = df.apply(lambda row: fix_position_skill_rating(row, 'RW'), axis=1)

    df['LAM'] = df.apply(lambda row: fix_position_skill_rating(row, 'LAM'), axis=1)
    df['CAM'] = df.apply(lambda row: fix_position_skill_rating(row, 'CAM'), axis=1)
    df['RAM'] = df.apply(lambda row: fix_position_skill_rating(row, 'RAM'), axis=1)

    df['LM'] = df.apply(lambda row: fix_position_skill_rating(row, 'LM'), axis=1)
    df['LCM'] = df.apply(lambda row: fix_position_skill_rating(row, 'LCM'), axis=1)
    df['CM'] = df.apply(lambda row: fix_position_skill_rating(row, 'CM'), axis=1)
    df['RCM'] = df.apply(lambda row: fix_position_skill_rating(row, 'RCM'), axis=1)
    df['RM'] = df.apply(lambda row: fix_position_skill_rating(row, 'RM'), axis=1)

    df['LWB'] = df.apply(lambda row: fix_position_skill_rating(row, 'LWB'), axis=1)
    df['LDM'] = df.apply(lambda row: fix_position_skill_rating(row, 'LDM'), axis=1)
    df['CDM'] = df.apply(lambda row: fix_position_skill_rating(row, 'CDM'), axis=1)
    df['RDM'] = df.apply(lambda row: fix_position_skill_rating(row, 'RDM'), axis=1)
    df['RWB'] = df.apply(lambda row: fix_position_skill_rating(row, 'RWB'), axis=1)

    df['LB'] = df.apply(lambda row: fix_position_skill_rating(row, 'LB'), axis=1)
    df['LCB'] = df.apply(lambda row: fix_position_skill_rating(row, 'LCB'), axis=1)
    df['CB'] = df.apply(lambda row: fix_position_skill_rating(row, 'CB'), axis=1)
    df['RCB'] = df.apply(lambda row: fix_position_skill_rating(row, 'RCB'), axis=1)
    df['RB'] = df.apply(lambda row: fix_position_skill_rating(row, 'RB'), axis=1)

    df = df.drop(columns=['Nationality', 'Club', 'Work Rate', 'Position', 'Loaned From', 'Body Type'])

    df.to_csv('D:\\Colin\\Programming\\DataSets\\fifa19\\groomed_data.csv', index=False)
    return df


def cluster_data(df):
    names = df['Name']

    # Remove goal keepers, they are a known separate cluster
    df = df.loc[df['Position ID'] != 5]
    df = df.drop(columns=['Name', 'ID', 'Value', 'Wage', 'Special', 'Real Face', 'Jersey Number',
                          'Joined', 'Contract Valid Until', 'Height', 'Weight', 'Nationality ID',
                          'Club ID', 'Body Type ID', 'Release Clause'])
    print('Irrelevant values dropped')
    # band_est = estimate_bandwidth(df) # -> 1
    band_est = 68
    print('Bandwidth:', band_est)
    analyzer = MeanShift(bandwidth=band_est)
    print('Fitting...')
    analyzer.fit(df)

    labels = analyzer.labels_

    df['Cluster Group'] = np.nan
    data_length = len(df)
    for i in range(data_length):
        df.iloc[i, df.columns.get_loc('Cluster Group')] = labels[i]

    df_clusters = df.groupby(['Cluster Group']).mean()
    df_clusters['Counts'] = pd.Series(df.groupby(['Cluster Group']).size())

    defense = df.loc[df['Cluster Group'] == 0]
    print(defense.describe().to_string())
    print()

    offense = df.loc[df['Cluster Group'] == 1]
    print(offense.describe().to_string())

    df['Name'] = names
    return df
