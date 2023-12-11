from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import numpy as np
import catboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
import plotly.express as px


class Airbnb():
    
    def __init__(self, df_public, df_privat, df_usd_rub, df_eur_rub, df_geo_city):
        self.df_usd_rub, self.df_eur_rub, self.df_geo_city = df_usd_rub, df_eur_rub, df_geo_city
        self.id_for_remove = pd.read_csv("id_for_remove.txt", sep=',')
        self.th = {
            "price_usd": {
                "min": 10,
                "max": 1_500
            }
        }
        self.models = []
        self.df_public, self.df_privat = df_public, df_privat
        self.df_train, self.df_test = train_test_split(self.df_public, test_size=0.2, random_state=20)
        self.X_public_prepared, self.X_privat_prepared, self.X_train_prepared, self.X_test_prepared = None, None, None, None
        self.X_public_encoded, self.X_privat_encoded, self.X_train_encoded, self.X_test_encoded = None, None, None, None
        self.pred_privat = None
        
    def get_preparation_date(self):
        self.X_public_prepared = self.preparation_data(self.df_public.copy(), message="df_public")
        self.X_privat_prepared = self.preparation_data(self.df_privat.copy(), message="df_privat", privat=True)
        self.X_train_prepared = self.preparation_data(self.df_train.copy(), message="df_train")
        self.X_test_prepared = self.preparation_data(self.df_test.copy(), message="df_test", train=False)
        return self.X_public_prepared, self.X_privat_prepared, self.X_train_prepared, self.X_test_prepared
    
    
    def get_encoded_data(self):
        if (self.X_public_prepared is None) & (self.X_privat_prepared is None) & (self.X_train_prepared is None) & (self.X_test_prepared is None):
            self.X_public_prepared, self.X_privat_prepared, self.X_train_prepared, self.X_test_prepared = self.get_preparation_date()
        self.X_public_encoded = self.encoding(self.X_public_prepared.copy(), message="X_public_prepared")
        self.X_privat_encoded = self.encoding(self.X_privat_prepared.copy(), message="X_privat_prepared")
        self.X_train_encoded = self.encoding(self.X_train_prepared.copy(), message="X_train_prepared")
        self.X_test_encoded = self.encoding(self.X_test_prepared.copy(), message="X_test_prepared")
        return self.X_public_encoded, self.X_privat_encoded, self.X_train_encoded, self.X_test_encoded
    
    
    def preparation_data(self, df, message="", train=True, privat=False):
        print(f"{message} start preparation - ", end="")
        df["id"] = df["id"].astype(str)
        df = df.rename(columns={'city': 'region', 
                                'neighbourhood_cleansed': 'city'})
        if not privat:
            df["price_usd"] = df.apply(lambda row: self.change_exchange_rate(row, self.df_usd_rub, self.df_eur_rub), axis=1)
        if train and not privat:
            df = self.cleaning_data(df, self.th, self.id_for_remove)
        df["host_is_superhost"].fillna('f', inplace=True)
        df["host_since"] = pd.to_datetime(df["host_since"])
        df["calendar_last_scraped"] = pd.to_datetime(df["calendar_last_scraped"])
        df["first_review"] = pd.to_datetime(df["first_review"])
        df["last_review"] = pd.to_datetime(df["last_review"])
        df["first_review"] = df["first_review"].fillna(df["first_review"].median())
        df["last_review"] = df["last_review"].fillna(df["last_review"].median())
        df["difference_review"] = df["last_review"] - df["first_review"]
        df["difference_review"] = df["difference_review"].apply(lambda x: x.days)
        df['host_response_rate'] = pd.to_numeric(df['host_response_rate'].str.rstrip('%'))
        df["host_response_rate"].fillna(df["host_response_rate"].median(), inplace=True)        
        df['host_acceptance_rate'] = pd.to_numeric(df['host_acceptance_rate'].str.rstrip('%'))
        df["host_acceptance_rate"].fillna(df["host_acceptance_rate"].median(), inplace=True)
        df['host_verifications'] = df['host_verifications'].astype(str)
        df["distance_to_centre_city"] = df[["region", "latitude", "longitude"]].apply(lambda row: self.define_distance(row, self.df_geo_city), axis=1)
        conditions = [
            (df["distance_to_centre_city"] >= 0) & (df["distance_to_centre_city"] < 3),
            (df["distance_to_centre_city"] >= 3) & (df["distance_to_centre_city"] < 100),
            (df["distance_to_centre_city"] >= 100)
        ]
        values = ['from 0 to 3', 'from 3 to 100', 'from 300']
        df["cat_distance"] = np.select(conditions, values)
        df["amenities"] = df["amenities"].apply(self.remove_sumbol)
        df["len_description"] = df["description"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        bathrooms = df["bathrooms_text"]
        bathrooms = bathrooms.replace({"Half-bath": 0.5, "Shared half-bath": 0.5, "Private half-bath": 0.5})
        bathrooms = bathrooms.astype(str).str.extract('(\d+\.\d+|\d+)', expand=False)
        bathrooms = pd.to_numeric(bathrooms, errors='coerce')
        bathrooms = bathrooms.fillna(bathrooms.median())
        accommodates = pd.to_numeric(df["accommodates"])
        bedrooms = pd.to_numeric(df["bedrooms"], errors="coerce")
        bedrooms = bedrooms.fillna(bedrooms.median())
        beds = pd.to_numeric(df["beds"], errors="coerce")
        beds = beds.fillna(beds.median())
        df["bathrooms_num"] = bathrooms
        df["bedrooms_num"] = bedrooms
        df["beds_num"] = beds
        df["bathrooms_num2"] = bathrooms ** 2
        df["bedrooms_num2"] = bedrooms ** 2
        df["beds_num2"] = beds ** 2
        df_sup = pd.DataFrame({
            "accommodates": accommodates,
            "bathrooms": bathrooms,
            "bedrooms": bedrooms,
            "beds": beds
        })
        df["person_per_bathrooms"] = df_sup["bathrooms"] / df_sup["accommodates"]
        df["person_per_bedrooms"] = df_sup["bedrooms"] / df_sup["accommodates"]
        df["person_per_beds"] = df_sup["beds"] / df_sup["accommodates"]
        df["person_per_bathrooms2"] = df["person_per_bathrooms"] **2
        df["person_per_bedrooms2"] = df["person_per_bedrooms"] **2
        df["person_per_beds2"] = df["person_per_beds"] **2
        print("done")
        return df
    
    
    @staticmethod
    def cleaning_data(df, th, id_for_remove):
        # df = df[~df["id"].isin(id_for_remove)]
        df = df[(df["price_usd"] > th["price_usd"]["min"]) & (df["price_usd"] < th["price_usd"]["max"])]
        df = df.dropna(subset=["host_since", "host_is_superhost"])
        df = df.drop(df[df["minimum_nights"] > 365].index)
        df = df.drop(df[df["maximum_minimum_nights"] > 365].index)
        return df
    
    
    @staticmethod
    def encoding(df, message=""):
        print(f"{message} start encoding - ", end="")
        df_bathrooms_num2 = df["bathrooms_num2"]
        df_bedrooms_num2 = df["bedrooms_num2"]
        df_beds_num2 = df["beds_num2"]
        df_person_per_bathrooms2 = df["person_per_bathrooms2"]
        df_person_per_bedrooms2 = df["person_per_bedrooms2"]
        df_person_per_beds2 = df["person_per_beds2"]
        df_bathrooms_num = df["bathrooms_num"]
        df_bedrooms_num = df["bedrooms_num"]
        df_beds_num = df["beds_num"]
        df_person_per_bathrooms = df["person_per_bathrooms"]
        df_person_per_bedrooms = df["person_per_bedrooms"]
        df_person_per_beds = df["person_per_beds"]
        df_difference_review = df["difference_review"]
        df_first_review_year = df["first_review"].dt.year.astype(str).fillna("3000.0") #cat
        df_last_review_year = df["last_review"].dt.year.astype(str).fillna("3000.0") #cat
        df_len_description = df["len_description"]
        df_latitude = df["latitude"]
        df_longitude = df["longitude"]
        df_description = df["description"].fillna("null")
        df_amenities = df["amenities"]
        df_cat_distance = df["cat_distance"] #cat
        df_distance_to_centre_city = df["distance_to_centre_city"]
        df_source = df["source"] #cat
        df_host_since = df["host_since"].dt.year.astype(str) #cat
        df_host_response_time = df["host_response_time"].astype(str).fillna('missing') #cat
        df_host_response_rate = df["host_response_rate"].fillna(100_000).astype(int)
        df_host_acceptance_rate = df["host_acceptance_rate"].fillna(100_000).astype(int)
        df_host_is_superhost = df["host_is_superhost"].fillna('f').replace('nan', 'f').astype(str) #cat
        df_host_listings_count = df["host_listings_count"].fillna(100_000).astype(int)
        df_host_total_listings_count = df["host_total_listings_count"].fillna(100_000).astype(int)
        df_host_verifications = df["host_verifications"].astype(str)
        df_host_has_profile_pic = df["host_has_profile_pic"].astype(str).fillna("f") #cat
        df_host_identity_verified = df["host_identity_verified"].astype(str).fillna("f") #cat
        df_city = df["city"] # cat
        df_property_type = df["property_type"] #cat
        df_room_type = df["room_type"] #cat
        df_accommodates = df["accommodates"].astype(str) #cat
        df_bathrooms_text = df["bathrooms_text"].astype(str).fillna(100_000) #cat
        df_bedrooms = df['bedrooms'].replace({'one': 1, 'two': 2, 'three': 3}).astype(str) #cat
        df_beds = df['beds'].astype(str) #cat
        df_minimum_nights = df["minimum_nights"]
        df_minimum_minimum_nights = df["minimum_minimum_nights"].fillna(100_000).astype(int)
        df_minimum_nights_avg_ntm = df["minimum_nights_avg_ntm"].replace([np.inf, -np.inf], np.nan).fillna(100_000).astype(int)
        df_has_availability = df["has_availability"] # cat
        df_availability_30 = df["availability_30"]
        df_availability_60 = df["availability_60"]
        df_availability_90 = df["availability_90"]
        df_availability_365 = df["availability_365"]
        df_number_of_reviews = df["number_of_reviews"]
        df_number_of_reviews_ltm = df["number_of_reviews_ltm"].astype(int)
        df_number_of_reviews_l30d = df["number_of_reviews_l30d"].astype(int)
        df_review_scores_rating = df["review_scores_rating"].fillna(100_000).astype(int)
        df_review_scores_accuracy = df["review_scores_accuracy"].fillna(100_000).astype(int)
        df_review_scores_cleanliness = df["review_scores_cleanliness"].fillna(100_000).astype(int)
        df_review_scores_checkin = df["review_scores_checkin"].fillna(100_000).astype(int)
        df_review_scores_communication = df["review_scores_communication"].fillna(100_000).astype(int)
        df_review_scores_location = df["review_scores_location"].fillna(100_000).astype(int)
        df_review_scores_value = df["review_scores_value"].fillna(100_000).astype(int)
        df_instant_bookable = df["instant_bookable"] # cat
        df_calculated_host_listings_count = df["calculated_host_listings_count"].astype(int)
        df_calculated_host_listings_count_entire_homes = df["calculated_host_listings_count_entire_homes"].astype(int)
        df_calculated_host_listings_count_private_rooms = df["calculated_host_listings_count_private_rooms"].astype(int)
        df_calculated_host_listings_count_shared_rooms = df["calculated_host_listings_count_shared_rooms"].astype(int)
        df_reviews_per_month = df["reviews_per_month"].fillna(100_000).astype(int)
        df_region = df["region"] # cat
        
        df = pd.concat([df_region, df_reviews_per_month, df_calculated_host_listings_count_shared_rooms, 
                        df_calculated_host_listings_count_private_rooms, df_calculated_host_listings_count_entire_homes, 
                        df_calculated_host_listings_count, df_instant_bookable, df_review_scores_value, df_review_scores_location, 
                        df_review_scores_communication, df_review_scores_checkin, df_review_scores_cleanliness, df_review_scores_accuracy,
                        df_review_scores_rating,df_number_of_reviews_l30d, df_number_of_reviews_ltm, df_number_of_reviews,
                        df_availability_365, df_availability_90, df_availability_60, df_availability_30, df_has_availability, 
                        df_minimum_nights_avg_ntm, df_minimum_minimum_nights, df_minimum_nights, df_beds, df_bedrooms, df_bathrooms_text, 
                        df_accommodates, df_room_type, df_property_type, df_city, df_host_identity_verified, 
                        df_host_has_profile_pic, df_host_verifications, df_host_total_listings_count, df_host_listings_count, 
                        df_host_is_superhost, df_host_response_rate,  df_host_acceptance_rate, df_host_response_time, df_host_since, df_source, 
                        df_distance_to_centre_city, df_cat_distance, df_latitude, df_longitude, df_description, df_amenities, df_len_description,
                        df_first_review_year, df_last_review_year, df_difference_review, df_person_per_bathrooms,
                        df_person_per_bedrooms, df_person_per_beds, df_bathrooms_num, df_bedrooms_num, df_beds_num,
                        df_person_per_beds2, df_person_per_bedrooms2, df_person_per_bathrooms2, df_beds_num2,
                        df_bedrooms_num2, df_bathrooms_num2], axis=1)
        print("done")
        return df
    
    
    def fit(self, public=False):
        cat_features = [0, 6, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 37, 40, 42, 44, 50, 51]
        text_features = [47, 48]
        cgb = catboost.CatBoostRegressor(cat_features=cat_features, 
                                         text_features=text_features, 
                                         verbose=100)
        if public:
            train_pool = catboost.Pool(data=self.X_public_encoded, label=self.X_public_prepared["price_usd"], cat_features=cat_features, text_features=text_features)
            cgb.fit(train_pool)
        else:
            train_pool = catboost.Pool(data=self.X_train_encoded, label=self.X_train_prepared["price_usd"], cat_features=cat_features, text_features=text_features)
            cgb.fit(train_pool)
        self.models.append(cgb)

        feature_importance = cgb.feature_importances_
        feature_names = cgb.feature_names_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importance
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            title='Feature Importance',
            labels={'Importance': 'SHAP Values'},
            height=1000
        )
        fig.show()
     
    
    def eval_model(self):
        y_pred_train, y_pred_test = [], []
        for model in self.models:
            y_pred_train = model.predict(self.X_train_encoded)
            y_pred_test = model.predict(self.X_test_encoded)
        mae_train = mean_absolute_error(self.X_train_prepared["price_usd"], y_pred_train)
        mae_test = mean_absolute_error(self.X_test_prepared["price_usd"], y_pred_test)
        r2_train = r2_score(self.X_train_prepared["price_usd"], y_pred_train)
        r2_test = r2_score(self.X_test_prepared["price_usd"], y_pred_test)
        return mae_train, mae_test, r2_train, r2_test
        

    def predict(self):
        for model in self.models:
            self.pred_privat = model.predict(self.X_privat_encoded)
        df_sub = pd.DataFrame({ "id": self.df_privat["id"], "price": self.pred_privat})
        return df_sub
    
    
    @staticmethod
    def change_exchange_rate(df, df_usd_rub, df_eur_rub):
        date = pd.to_datetime(df["calendar_last_scraped"])
        price = df["price"]
        
        if price[0] == '$':
            usd = float(price[1:].replace(',', ''))
            return usd
        
        if price[0] == '€':
            eur = float(price[1:].replace(',', ''))
            closest_date_eur_rub = df_eur_rub["data"].iloc[(df_eur_rub["data"] - date).abs().argsort()[0]]
            exchange_rate_eur_rub = float(df_eur_rub.loc[df_eur_rub["data"] == closest_date_eur_rub, "curs"])
            rub = eur * exchange_rate_eur_rub
            closest_date_usd_rub = df_usd_rub["data"].iloc[(df_usd_rub["data"] - date).abs().argsort()[0]]
            exchange_rate_usd_rub = float(df_usd_rub.loc[df_usd_rub["data"] == closest_date_usd_rub, "curs"])
            usd = rub / exchange_rate_usd_rub
            return usd
            
        if price[0] == '₽':
            rub = float(price[1:].replace(',', ''))
            closest_date_usd_rub = df_usd_rub["data"].iloc[(df_usd_rub["data"] - date).abs().argsort()[0]]
            exchange_rate__usd_rub = float(df_usd_rub.loc[df_usd_rub["data"] == closest_date_usd_rub, "curs"])
            usd = rub / exchange_rate__usd_rub
            return usd 
        
    
    @staticmethod
    def define_distance(row, df_geo_city):
        region, latitude_home, longitude_home = row["region"], row['latitude'], row['longitude']
        geo = df_geo_city[df_geo_city["region"] == region][["latitude_city", "longitude_city"]].values[0]
        latitude_city, longitude_city = geo[0], geo[1]
        
        R = 6371.0
        lat1_rad = radians(latitude_home)
        lon1_rad = radians(longitude_home)
        lat2_rad = radians(latitude_city)
        lon2_rad = radians(longitude_city)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance
    
    @staticmethod
    def remove_sumbol(s):
        translation_table = str.maketrans("", "", '[]"')
        return s.translate(translation_table)