import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Cleaner :

    __dataset_path = "./data/Kangaroo.csv"

    @staticmethod
    def load_data(self, path: str) -> pd.DataFrame :
        '''
            Load the dataset
            :return: DataFrame
        '''
        return pd.read_csv(path)
    
    @staticmethod
    def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Remove duplicates from the dataset
            :param data: DataFrame to clean
            :return: cleaned DataFrame
            Rows:
                Indexes 5591 - 440319 is duplicate id: 20663057.0
        '''
        data.drop_duplicates(subset=['id'], keep='last', inplace=True)
        return data
    
    @staticmethod
    def drop_na(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Remove rows with NaN values in price column
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        data.dropna(subset=['price'], how='all', inplace=True)
        return data
    
    @staticmethod
    def drop_columns(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Remove columns from the dataset
            :param data: DataFrame to clean
            :return: cleaned DataFrame
            Columns:
                0 index
                1 id
                2 url
                # 4 subtype
                11 roomCount
                12 monthlyCost
                34 hasBalcony
                35 hasGarden
                # 38 parkingCountIndoor
                # 39 parkingCountOutdoor
                50 accessibleDisabledPeople
        '''
        data.drop(data.columns[[0, 1, 2, 11, 12, 34, 35, 50]], axis=1, inplace=True)
        return data

    @staticmethod
    def clean_epcScore(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Replace strange values in epcScore column with NaN
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        values_to_replace = ['C_A', 'F_C', 'G_C', 'D_C', 'F_D', 'E_C', 'G_E', 'E_D', 'C_B', 'X', 'G_F']
        data.loc[data['epcScore'].isin(values_to_replace), 'epcScore'] = np.nan
        return data
    
    @staticmethod
    def replace_outlier_toiletCount(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Replace outlier value in toiletCount column with 2
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        data.loc[data['toiletCount'] == 1958, 'toiletCount'] = 2
        return data 
    
    @staticmethod
    def float_to_int(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Convert columns to integer
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        to_int = lambda x : int(x)
        # 'id', 'roomCount'
        cols_to_int = ['bedroomCount', 'bathroomCount',
                        'habitableSurface', 'diningRoomSurface', 'kitchenSurface',
                        'landSurface', 'livingRoomSurface', 'gardenSurface',
                        'terraceSurface', 'buildingConstructionYear', 'facedeCount',
                        'floorCount', 'toiletCount']
        for col in cols_to_int :
            data[col] = data[col].fillna(data[col].median()).apply(to_int)
        return data
    
    @staticmethod
    def bool_to_int(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Convert boolean columns to integer
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        to_bool = lambda x : False if np.isnan(x) else x
        bool_to_int = lambda x : 1 if x == True else 0
        # 'hasGarden'
        cols_to_bool = ['hasAttic', 'hasBasement', 'hasDressingRoom',
                        'hasDiningRoom', 'hasLift', 'hasHeatPump',
                        'hasPhotovoltaicPanels', 'hasThermicPanels',
                        'hasLivingRoom', 'hasAirConditioning', 'hasArmoredDoor',
                        'hasVisiophone', 'hasOffice', 'hasSwimmingPool',
                        'hasFireplace', 'hasTerrace']
        for col in cols_to_bool :
            # mode_val = data[col].isnull().count() .apply(to_bool).mode()[0]
            data[col] = data[col].apply(bool_to_int)
        return data
    
    @staticmethod
    def round_float(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Round float columns to 2 decimal places
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        round_float = lambda x : round(float(x), 2)
        
        cols_to_round = ['streetFacadeWidth']
        for col in cols_to_round :
            data[col] = data[col].fillna(data[col].median()).apply(round_float)
        return data
    
    @staticmethod
    def type_to_int(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Replace type column with 0 if APARTMENT, 1 if HOUSE
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        type_to_int = lambda x : 0 if x == 'APARTMENT' else 1

        data['type'] = data['type'].apply(type_to_int)
        return data

    @staticmethod
    def flood_zone_nan_to_non_food_zone(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Replace missing values in flood zone column with NON_FLOOD_ZONE
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        flood_zone_replace_nan = lambda x : 'NON_FLOOD_ZONE' if pd.isnull(x) else x
        data.floodZoneType = data.floodZoneType.apply(flood_zone_replace_nan)
        return data
    
    @staticmethod
    def clean_kitchen_type(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Clean kitchen type column
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        kitchen_type_cleaner = lambda x : 0 if pd.isnull(x) or "NOT" in x or "UN" in x else 0.5 if "SEMI" in x else 1

        data['kitchenType'] = data['kitchenType'].apply(kitchen_type_cleaner)
        return data
    
    @staticmethod
    def treat_dictionaries(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Treat 'dictionaries' in the dataset
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        # 'buildingCondition', , 'terraceOrientation', 'epcScore'
        cols = ['floodZoneType', 'heatingType', 'gardenOrientation']
        # 'condition', , 'terrace', 'epcScore'
        prefixes = ['', 'heating', 'garden']

        data = pd.get_dummies(data, columns=cols, prefix = prefixes, dtype=int)
        # Remove NON_FLOOD_ZONE column as we have replaced NaNs with NON_FLOOD_ZONE before)
        data.drop(columns=['_NON_FLOOD_ZONE'], inplace=True)
        # 'condition_AS_NEW',         #6
        # 'condition_GOOD',           #5
        # 'condition_JUST_RENOVATED', #4
        # 'condition_TO_BE_DONE_UP',  #3
        # 'condition_TO_RENOVATE',    #2
        # 'condition_TO_RESTORE',     #1
        data['buildingCondition'] = data['buildingCondition'].apply(lambda x : 0 if pd.isnull(x) else 
                                        5 if 'AS_NEW' in x else
                                        4 if 'GOOD' in x else
                                        4 if 'JUST_RENOVATED' in x else
                                        3 if 'TO_BE_DONE_UP' in x else
                                        2 if 'TO_RENOVATE' in x else 1)
        # 'epcScore_A',               #9
        # 'epcScore_A+',              #8
        # 'epcScore_A++',             #7
        # 'epcScore_B',               #6   
        # 'epcScore_C',               #5
        # 'epcScore_D',               #4
        # 'epcScore_E',               #3
        # 'epcScore_F',               #2
        # 'epcScore_G',               #1
        data['epcScore'] = data['epcScore'].apply(lambda x : 0 if pd.isnull(x) else 
                                            7 if 'A' == x else
                                            8 if 'A+' == x else
                                            9 if 'A++' == x else
                                            6 if 'B' == x else
                                            5 if 'C' == x else 
                                            4 if 'D' == x else
                                            3 if 'E' == x else
                                            2 if 'F' == x else 1)
        # 'SOUTH'         # 8
        # 'SOUTH_EAST'    # 7     
        # 'SOUTH_WEST'    # 6      
        # 'EAST'          # 5
        # 'WEST'          # 4
        # 'NORTH_EAST'    # 3
        # 'NORTH_WEST'    # 2
        # 'NORTH'         # 
        data['terraceOrientation'] = data.apply(lambda row : 0 if pd.isnull(row.terraceOrientation) else
                                        8 if 'SOUTH' in row.terraceOrientation else
                                        7 if 'SOUTH_EAST' in row.terraceOrientation else
                                        6 if 'SOUTH_WEST' in row.terraceOrientation else
                                        5 if 'EAST' in row.terraceOrientation else
                                        4 if 'WEST' in row.terraceOrientation else
                                        3 if 'NORTH_EAST' in row.terraceOrientation else
                                        2 if 'NORTH_WEST' in row.terraceOrientation else 1, axis = 1)
        return data
        
    @staticmethod
    def create_region_column(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Create region column based on province
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        wallonie_provs = ['Luxembourg', 'Liège', 'Walloon Brabant', 'Namur' 'Hainaut']              # 1
        flandre_provs = ['West Flanders', 'East Flanders', 'Antwerp', 'Flemish Brabant', 'Limburg'] # 2
        # 'Brussels'                                                                                # 3

        data['region'] = data['province'].apply(lambda x : 2 if x in flandre_provs else 
                                              1 if x in wallonie_provs else 3)
        # self.data.drop(columns=['province', 'locality'], inplace=True) #
        return data
    
    @staticmethod
    def locality_to_upper(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Convert locality column to uppercase
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        data['locality'] = data['locality'].str.upper()
        return data
    
    @staticmethod
    def get_rid_of_outliers_col(data: pd.DataFrame, column_name: str) -> pd.DataFrame :
        '''
            Remove outliers from the dataset
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        # Remove outliers from the column
        q1 = data[column_name].quantile(0.25)
        q3 = data[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
        return data
    
    @staticmethod
    def one_hot_encode(data: pd.DataFrame) -> pd.DataFrame :
        s = (data.dtypes == 'object')
        object_cols = list(s[s].index)
        print(f"Categorical variables: {object_cols}")
        print(f'No. of. categorical features: {len(object_cols)}')
        OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        OH_cols = pd.DataFrame(OH_encoder.fit_transform(data[object_cols]))
        OH_cols.index = data.index
        OH_cols.columns = OH_encoder.get_feature_names_out()
        data = data.drop(object_cols, axis=1)
        data = pd.concat([data, OH_cols], axis=1)
        return data

    @staticmethod
    def clean_data() -> pd.DataFrame :
        return (
                pd.DataFrame().pipe(Cleaner.load_data, Cleaner.__dataset_path)
                .pipe(Cleaner.drop_duplicates)
                .pipe(Cleaner.drop_na)
                .pipe(Cleaner.get_rid_of_outliers_col, 'price')
                .pipe(Cleaner.get_rid_of_outliers_col, 'habitableSurface')
                .pipe(Cleaner.drop_columns)
                .pipe(Cleaner.clean_epcScore)
                .pipe(Cleaner.replace_outlier_toiletCount)
                .pipe(Cleaner.float_to_int)
                .pipe(Cleaner.bool_to_int)
                .pipe(Cleaner.round_float)
                .pipe(Cleaner.locality_to_upper)
            )
        # #self.flood_zone_nan_to_non_food_zone()
        # #self.clean_kitchen_type()
        # #self.treat_dictionaries()
        # #self.create_region_column()
        # #self.one_hot_encode()
        # return self.data