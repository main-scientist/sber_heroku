import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template

file_id ="1YWbCavIigITOm5EtaE84CN0t1xhEmW9X"
url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
url = 'https://drive.google.com/uc?export=download&confirm=1&id='+url.split('/')[-2]
X_public = pd.read_csv(url)

df_geo_city = pd.DataFrame({
    "region": ["Bergamo", "Bologna", "Firenze", "Milano", "Napoli", "Puglia", "Roma", "Sicilia", "Trentino", "Venezia"],
    "latitude_city": [45.696, 44.4938, 43.7793, 45.4643, 40.8522, 41.1175800, 41.8919, 37.50788 , 46.0679, 45.4371],
    "longitude_city": [9.66721, 11.3387, 11.2463, 9.18951, 14.2681, 16.4842100, 12.5113, 15.08303, 11.1211, 12.3327]
})
load_figure_template(["cyborg", "darkly"])
X_public['latitude'] = pd.to_numeric(X_public['latitude'], errors='coerce')
X_public['longitude'] = pd.to_numeric(X_public['longitude'], errors='coerce')
mean_latitude = X_public['latitude'].mean()
mean_longitude = X_public['longitude'].mean()


# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
#                 meta_tags=[{'name': 'viewport',
#                             'content': 'width=device-width, initial-scale=1.0'}]
#                   )

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

server = app.server

dark_theme = {
    "main-background": "#000000",
    "header-text": "#ff7575",
    "sub-text": "#eee",
}


# Graph
heatmap_data = pd.isnull(X_public)
heatmap_fig = px.imshow(heatmap_data, 
                        labels=dict(color="Null Values"), 
                        color_continuous_scale='Viridis', title="Number of null value", template='darkly')
heatmap_fig.update_layout(
    annotations=[
        dict(
            text="Number of duplicate rows: 0",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=1.07,
            font=dict(size=16),
        )
    ],
    height=800
)
median_price = X_public["price_usd"].median()
mean_price = X_public["price_usd"].mean()
price_fig = px.histogram(X_public, x="price_usd", marginal="box", title="Price (USD) with Median", template='darkly')
price_fig.add_shape(
    go.layout.Shape(
        type='line',
        x0=median_price,
        x1=median_price,
        y0=0,
        y1=1,
        xref='x',
        yref='paper',
        line=dict(color='red', width=2)
    )
)
price_fig.add_shape(
    go.layout.Shape(
        type='line',
        x0=mean_price,
        x1=mean_price,
        y0=0,
        y1=1,
        xref='x',
        yref='paper',
        line=dict(color='blue', width=2)
    )
)
price_fig.update_layout(
    annotations=[
        dict(
            x=median_price,
            y=1.02,
            xref='x',
            yref='paper',
            text=f'Median: {median_price:.2f}',
            showarrow=True,
            arrowhead=2,
            arrowcolor='red',
            ax=0,
            ay=-40
        ),
        dict(
            x=mean_price,
            y=1.05,
            xref='x',
            yref='paper',
            text=f'Mean: {mean_price:.2f}',
            showarrow=True,
            arrowhead=2,
            arrowcolor='blue',
            ax=0,
            ay=-40
        ),
    ]
)


price_fig.update_layout(xaxis_title_text='Price (USD)')

summary_statistics = X_public['price_usd'].describe()


histogram_sourse = px.histogram(X_public["source"], title="Histogram of Source", template='darkly')
histogram_host_response_time = px.histogram(X_public["host_response_time"], title="Histogram of Host response time", template='darkly')
histogram_host_response_rate = px.histogram(X_public["host_response_rate"], title="Histogram of Host response rate", template='darkly')
histogram_host_acceptance_rate = px.histogram(X_public["host_acceptance_rate"], title="Histogram of Host acceptance rate", template='darkly')
histogram_host_is_superhost = px.histogram(X_public["host_is_superhost"], title="Histogram of Host is superhost", template='darkly')
histogram_host_listings_count = px.histogram(X_public["host_listings_count"], title="Histogram of Host listings count", template='darkly')
histogram_host_total_listings_count = px.histogram(X_public["host_total_listings_count"], title="Histogram of Host total listings count", template='darkly')
histogram_host_verifications = px.histogram(X_public["host_verifications"], title="Histogram of Host verifications", template='darkly')
histogram_accommodates = px.histogram(X_public["accommodates"], title="Histogram of Accommodates", template='darkly')
histogram_number_of_reviews = px.histogram(X_public["number_of_reviews"], title="Histogram of Number of reviews", template='darkly')
histogram_host_has_profile_pic = px.histogram(X_public["host_has_profile_pic"], title="Histogram of Number of reviews", template='darkly')
histogram_distance_to_centre_city = px.histogram(X_public["distance_to_centre_city"], title="Histogram of distance to the centre of city", template='darkly')
histogram_cat_distance = px.histogram(X_public["cat_distance"], title="Histogram of cat distance", template='darkly')

densitymapbox_price = go.Figure(go.Densitymapbox(
    lat=X_public['latitude'],
    lon=X_public['longitude'],
    z=X_public['price_usd'],
    radius=15,
    colorbar=dict(
        title='Price (USD)',
        tickvals=[X_public['price_usd'].min(), X_public['price_usd'].max()],
    ),
))
densitymapbox_price.update_layout(
    mapbox=dict(
        center=dict(lat=mean_latitude, lon=mean_longitude),
        style='open-street-map',
        zoom=5,
    ),
    showlegend=False,
    margin=dict(l=15, r=15, t=50, b=15),
    height=600,
    template='plotly_dark',
    annotations=[
        dict(
            text="Price Distribution on Map by Coordinates",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.03,
            y=1.06,
            font=dict(size=16),
        )
    ],
)

# func

def generate_hypothesis_block(method, a, b, correlation_coefficient, p_value, significance_level, reject_null_hypothesis):
    return html.Div([
        dcc.Markdown(f'''
            #### Hypothesis
            H0 - Нет статистически значимой корреляции между **{a}** и **{b}**
            
            H1 - Существует статистически значимая корреляция между **{a}** и **{b}**.
            
            {method} correlation coefficient: `{correlation_coefficient}`
            
            P-value: `{p_value}`
            
            Level of Significance: `{significance_level}`
            
            {'Отклоняем' if reject_null_hypothesis else 'Не отклоняем'} нулевую гипотезу.
        '''),
        ], style={'margin-top': '60px', 'margin-left': '50px', 'margin-right': '60px', "background-color": "#303030", "padding": "20px", "border-radius":"10px", 'display': 'inline-block', 'width': '30%'})













app.layout = dbc.Container(children=[
    html.H2("В данный момент презентация находится в стадии разработки"),
    html.H2("Predicting Real Estate Prices from the Airbnb Portal"),
    html.H3("Content"),
    
    dcc.Markdown('''
    - [Explore Data Analysis](#explore-data-analysis)
    - [Data Preparation](#data-preparation)
    - [Feature Engineering](#feature-engineering)
    - [Modeling](#modeling)
        - [Comparative analysis of models](#comparative-analysis-models)
    ''', style={'position': 'fixed', 'top': 0, 'left': 0, 'background-color': '#000000', 'padding': '15px', 'z-index': 1000, "border-radius":"10px"}),
    
    html.H3("Explore Data Analysis", id="explore-data-analysis", style={'margin': '50px'}),

    html.H4("Data Frame - Public", style={'margin-left': '50px'}),
    
    dash_table.DataTable(
        id='table',
        columns=[
            {'name': col, 'id': col} for col in X_public.columns
        ],
        data=X_public.head(2).to_dict('records'),  # Display the first 2 rows
        style_table={'overflowX': 'auto', 'width': '100%'},
        style_cell={'maxWidth': 300, 'textAlign': 'center', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'color': 'white', 'backgroundColor': '#000'},
    ),
    
    dcc.Graph(
        id='heatmap',
        figure=heatmap_fig,
        style={'border': '2px solid black', "margin-bottom":"50px"}
    ),
    
    dcc.Markdown('''
        ### Target - Price
        Заметим, что денежные значения у целeвого признака представлены в разных валютах: '$' '€' '₽'.
        
        В наборе данных присутсвует признак `calendar_last_scraped` - "последние обновление данных". Эти даты будут использованы для конвертации валют по корректному курсу.
        
        Возьмем курсы валют с официального портала Банк России и конвертируем их.
    ''', style={'margin-left': '50px','fontSize': '18px'}),
    
    
    dcc.Graph(
        id='price_fig',
        figure=price_fig,
        style={'border': '2px solid black', "margin-bottom":"20px"}
    ),
    
    
    html.Div([
        dash_table.DataTable(
            id='summary-table-price',
            columns=[
                {'name': 'Statistic', 'id': 'statistic'},
                {'name': 'Value', 'id': 'value'}
            ],
            data=[
                {'statistic': 'Count', 'value': X_public['price_usd'].describe()['count']},
                {'statistic': 'Mean', 'value': X_public['price_usd'].describe()['mean']},
                {'statistic': 'Std', 'value': X_public['price_usd'].describe()['std']},
                {'statistic': 'Min', 'value': X_public['price_usd'].describe()['min']},
                {'statistic': '25%', 'value': X_public['price_usd'].describe()['25%']},
                {'statistic': '50% (Median)', 'value': X_public['price_usd'].describe()['50%']},
                {'statistic': '75%', 'value': X_public['price_usd'].describe()['75%']},
                {'statistic': 'Max', 'value': X_public['price_usd'].describe()['max']},
            ],
            style_table={'overflowX': 'auto', 'width': '300px', 'margin-left': '50px'},
            style_cell={'textAlign': 'center', 'overflow': 'hidden', 'textOverflow': 'ellipsis','color': 'white', 'backgroundColor': '#000'}
        ),
    ], style={'display': 'inline-block', 'width': '30%'}),
    
    html.Div([
        dcc.Markdown('''
            Применим медианный подход для центральной тенденции и примем решение об ограничении выборки, исключив значения, выходящие за пределы трех стандартных отклонений от медианы. 
            Это позволит учесть возможное влияние выбросов на статистическую оценку и обеспечит более устойчивое представление о центре распределения
        '''),
    ], style={'text-align': 'justify', 'display': 'inline-block', 'width': '50%', 'vertical-align': 'top', 'fontSize': '18px'}),
    
    
    generate_hypothesis_block("Biserial", "sourse", "price_usd", 0.01143, 0.00056, 0.05, True),
    dcc.Graph(
        id='histogram_source',
        figure=histogram_sourse,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ),
    
    generate_hypothesis_block("Biserial", "host_response_time", "price_usd", 0.00564, 0.08907, 0.05, False),
    dcc.Graph(
        id='histogram_host_response_time',
        figure=histogram_host_response_time,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ),
    
    generate_hypothesis_block("Pearson", "host_response_rate", "price_usd", -0.02782, 0.0, 0.05, True),
    dcc.Graph(
        id='histogram_host_response_rate',
        figure=histogram_host_response_rate,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ), 
    
    generate_hypothesis_block("Pearson", "host_acceptance_rate", "price_usd", 0.05246, 0.0, 0.05, True),
    dcc.Graph(
        id='histogram_host_acceptance_rate',
        figure=histogram_host_acceptance_rate,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ), 
    
    generate_hypothesis_block("Biserial", "host_is_superhost", "price_usd", 0.01452, 1e-05, 0.05, True),
    dcc.Graph(
        id='histogram_host_is_superhost',
        figure=histogram_host_is_superhost,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ), 

    generate_hypothesis_block("Pearson", "host_listings_count", "price_usd", 0.11413, 0.0, 0.05, True),
    dcc.Graph(
        id='histogram_host_listings_count',
        figure=histogram_host_listings_count,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ),
    generate_hypothesis_block("Pearson", "host_total_listings_count", "price_usd", 0.09509, 0.0, 0.05, True),
    dcc.Graph(
        id='histogram_host_total_listings_count',
        figure=histogram_host_total_listings_count,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ), 
    generate_hypothesis_block("Biserial", "host_verifications", "price_usd", 0.03171, 0.00056, 0.05, True),
    dcc.Graph(
        id='histogram_host_verifications',
        figure=histogram_host_verifications,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ), 
    generate_hypothesis_block("Pearson", "accommodates", "price_usd", 0.35318, 0.00056, 0.05, True),
    dcc.Graph(
        id='histogram_accommodates',
        figure=histogram_accommodates,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ),
    generate_hypothesis_block("Pearson", "number_of_reviews", "price_usd", -0.04753, 0.00056, 0.05, True),
    dcc.Graph(
        id='histogram_number_of_reviews',
        figure=histogram_number_of_reviews,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ),
    generate_hypothesis_block("Pearson", "host_has_profile_pic", "price_usd", 0.01902, 0.00056, 0.05, True),
    dcc.Graph(
        id='histogram_host_has_profile_pic',
        figure=histogram_host_has_profile_pic,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ),
    
    
    
    html.H3("Feature Engineering", id="feature-engineering", style={'margin': '60px'}),
    
    
    html.Div([
        dcc.Markdown('''
            Используем `latitude` и `longitude` для представления географических координат недвижимости в разных городах Италии.
        '''),
    ], style={'fontSize': '18px', 'margin-left': '50px'}),
    
    dcc.Graph(
        id='densitymapbox_price',
        figure=densitymapbox_price,
    ),
    
    
    html.Div([
        dcc.Markdown('''
            Извлекая информацию из общедоступного интернет-ресурса, я получил географические координаты (долготу и широту) городов Италии, упомянутых в предоставленных данных об объектах недвижимости. 
            Далее я вычислил расстояние от каждого объекта до центра города, в котором он расположен, и создал два новых признака для анализа.

            Первый признак - `distance_to_centre_city` - является непрерывным и отражает фактическое расстояние в километрах от объекта недвижимости до центра города. 

            Второй признак - `cat_distance` - является категориальным и представляет расстояние в определенных диапазонах. 
            Этот признак подразумевает три категории: (from 0 to 3, from 3 to 100, from 300), что позволяет упростить восприятие и анализ данных, разбивая объекты недвижимости на группы по их удаленности от центра города.
        '''),
    ], style={'text-align': 'justify', 'fontSize': '18px', 'margin-left': '50px', 'margin-top': '50px', 'margin-right': '70px', 'display': 'inline-block', 'width': '50%'}),
    
    html.Div([
        DataTable(
        id='geo-table',
        columns=[
            {'name': col, 'id': col} for col in df_geo_city.columns
        ],
        data=df_geo_city.to_dict('records'),
        style_cell={'color': 'white', 'backgroundColor': '#000', 'textAlign': 'center'},
    ),
    ], style={'display': 'inline-block', 'width': '30%', 'vertical-align': 'top', 'margin-top': '50px'}),
    
    
    generate_hypothesis_block("Pearson", "distance_to_centre_city", "price_usd", -0.09707, 0.0, 0.05, True),
    dcc.Graph(
        id='histogram_distance_to_centre_city',
        figure=histogram_distance_to_centre_city,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ),
    generate_hypothesis_block("Biserial", "cat_distance", "price_usd", -0.02601, 0.0, 0.05, True),
    dcc.Graph(
        id='histogram_cat_distance',
        figure=histogram_cat_distance,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ),
    
    
    html.Div([
        dcc.Markdown('''
            Признаки которые были сгенерериваны из имеющихся данный 
            
            * `difference_review` - разница между признаками `first_review` и `last_review`, пропущенные значения были заполнены медианой.
            * `len_description` - длина символов признака description.
            * `bathrooms_num2` - добавление квадратичного члена к признаку `bathrooms` для учета нелинейных взаимосвязей.
            * `bedrooms_num2`- добавление квадратичного члена к признаку `bedrooms` для учета нелинейных взаимосвязей.
            * `beds_num2` - добавление квадратичного члена к признаку `beds` для учета нелинейных взаимосвязей.
            * `person_per_bathrooms` - кол-во человек на ванную комнату.
            * `person_per_bedrooms`- кол-во человек на спальную комнату.
            * `person_per_beds` - кол-во человек на кровать.
            * Эти признаки (3 последних) могут давать представление о плотности проживания или комфорте для каждого измерения пространства.
        '''),
    ], style={'fontSize': '18px', 'margin-left': '50px', "margin-top": "20px"}),
    
    
    
    
    html.H3("Modeling", id="modeling", style={'margin-left': '50px', 'margin-top': '70px', 'margin-bottom': '40px'}),
    
    html.Div([
        dcc.Markdown('''
            В качестве модели был выбран градиентный бустинг, в качестве реализации была выбрана библиотека catboost.
            
            Почему ?
            
            CatBoost эффективно обрабатывает категориальные признаки `cat_features` "из коробки", используя метод кодирования, основанный на изменении значений таргета.
            
            CatBoost поддерживает работу с текстовыми данными.
            - Loading and storing text features
            - Tokenization
            - Dictionary creation
            - Converting strings to numbers (Each string from the text feature is converted to a token identifier from the dictionary)
            - Estimating numerical features (Numerical features are calculated based on the source tokenized)
        '''),
    ], style={'fontSize': '18px', 'margin-left': '50px', "width": "900px"}),
    
    html.Div([
        dcc.Markdown('''
           Для подбора гиперпараметров был выбран байесовский оптимизатор.
           
           Это метод использует байесовский подход для поиска глобального экстремума (минимума или максимума) неизвестной целевой функции. 
           
           Основная идея заключается в том, чтобы использовать вероятностную модель для целевой функции и выбирать следующие точки для оценки так, чтобы максимизировать ожидаемое улучшение результатов.
           
           В качестве реализации была выбрана библиотека [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization).
        '''),
    ], style={'fontSize': '18px', 'margin-left': '50px', "width": "900px", "margin-top": "40px"}),
    
    
    dcc.Markdown('''
        Код осуществляет оптимизацию гиперпараметров для модели CatBoostRegressor с использованием байесовской оптимизации. 
        Оптимизация проводится путем минимизации RMSE, оцененной на основе кросс-валидации с тремя фолдами.

        ```python
        def cat_eval(iterations, depth, learning_rate, l2_leaf_reg, min_data_in_leaf, cat_features, text_features):
        params = {
            'iterations': int(iterations),
            'depth': int(depth),
            'learning_rate': learning_rate,
            'l2_leaf_reg': l2_leaf_reg,
            'min_data_in_leaf': int(min_data_in_leaf),
            'loss_function': 'RMSE',
            'eval_metric': 'MAE',
            'cat_features': cat_features,
            'text_features': text_features,
            'verbose': False
        }
    
        cv_scores = cross_val_score(catboost.CatBoostRegressor(**params), 
                                    X_public_encoded_null.drop("price_usd", axis=1), y_public_null, cv=3, 
                                    scoring='neg_mean_squared_error')
        return cv_scores.mean()

        pbounds = {
            'iterations': (100, 4000),
            'depth': (4, 10),
            'learning_rate': (0.001, 0.1),
            'l2_leaf_reg': (1, 10),
            'min_data_in_leaf': (1, 40),
            "cat_features": cat_features,
            "text_features": text_features
        }

        cat_opt = BayesianOptimization(
            f=cat_eval,
            pbounds=pbounds
        )

        cat_opt.maximize(n_iter=4, init_points=3)
        ```''', style={'background-color': '#333', "padding-top": "10px"}),
    
    
    
    html.H4("Comparative analysis of models", id="comparative-analysis-models", style={'margin-left': '50px', "margin-top": "40px"}),
    
    html.Div([
        dcc.Markdown('''
            Я перепепробовал множество различных подходов моделирования:
            
            * Simple Gradient Boosting (CatBoost)
            * Gradient Boosting with BayesianOptimization (CatBoost)
            * Aggregation Gradient Boosting with BayesianOptimization (CatBoost)
            * Bagging over Gradient Boosting with BayesianOptimization (num_samples=7, 50% of the general population, CatBoost)
            
                Данный способ показал лучший результат "бизнес метрики", но я считаю данную модель некомпетентной. Об этом далее.
            
            * Two-level Gradient Boosting Gradient Boosting with BayesianOptimization (CatBoost)
            
                Разделение генеральной совокупности на две выборки по целевому признаку [μ - 3σ : μ + σ] | [μ + σ : μ + 3σ]

                Обучение классификатора CatBoost with BayesianOptimization.
                
                Данный способ показал лучший результат по метрике MAE.
                
            * Three-level Gradient Boosting Gradient Boosting with BayesianOptimization (CatBoost)
            * Four-level Gradient Boosting Gradient Boosting with BayesianOptimization (CatBoost)
        '''),
    ], style={'fontSize': '18px', 'margin-left': '50px', "width": "1200px", "margin-top": "40px"}),
    
    
    html.Div(["Loss Function and Target Selection Table"], style={'fontSize': '22px', 'margin-left': '50px', "margin-top": "40px"}),
    
    html.Div([
        dash_table.DataTable(
            id='loss-function-table',
            columns=[
                {'name': 'General Population Interval', 'id': 'interval'},
                {'name': 'Log Target', 'id': 'log_target'},
                {'name': 'Loss Function', 'id': 'loss_function'},
            ],
            data=[
                {'interval': '[μ - 3σ : μ + σ]', 'log_target': 'TRUE', 'loss_function': 'RMSE'},
                {'interval': '[μ + σ : μ + 5σ]', 'log_target': 'TRUE', 'loss_function': 'MAPE'},
                {'interval': '[μ + 5σ : +inf)', 'log_target': 'FALSE', 'loss_function': 'MAPE'},
            ],
            style_table={'overflowX': 'auto', 'width': '500px', 'margin-left': '50px'},
            style_cell={'textAlign': 'center', 'overflow': 'hidden', 'textOverflow': 'ellipsis','color': 'white', 'backgroundColor': '#000'}
        ),
        ],
        style={'display': 'inline-block', 'width': '50%'}
    ),
    
    html.Div(['''Я использовал логарифмирование целевого признака, так как это стабилизирует дисперсию и cнижает влияние выбросов.
              Использовние MAPE как функции потерь обусловлено обеспечением однородности в ошибках. Даже если фактические значения имеют разные порядки величин, 
              MAPE отражает точность прогноза в процентах и может быть более информативной в контексте моей задачи, а также менее чувствительной к выбросам.'''],
            style={'text-align': 'justify', 'display': 'inline-block', 'width': '40%', 'vertical-align': 'top', 'fontSize': '18px'}
            ),


    html.Div(["Comparison of Models Based on Log Target Intervals and Loss Functions"], style={'fontSize': '22px', 'margin-left': '50px', "margin-top": "50px", "margin-bottom": "10px"}),
    
    dash_table.DataTable(
        id='comparison-table',
        columns=[
            {'name': 'Interval', 'id': 'interval'},
            {'name': 'Two-level GB', 'id': 'two_level_gb'},
            {'name': 'Three-level GB', 'id': 'three_level_gb'},
            {'name': 'Four-level GB', 'id': 'four_level_gb'},
            {'name': 'Earned relative total value', 'id': 'earned_relative_total_value'},
        ],
        data=[
            {'interval': '[μ - 3σ : μ + σ]', 'two_level_gb': 'Log Target RMSE', 'three_level_gb': 'Log Target RMSE', 'four_level_gb': 'Log Target RMSE', 'earned_relative_total_value': '45%'},
            {'interval': '[μ + σ : μ + 3σ]', 'two_level_gb': 'Log Target MAPE', 'three_level_gb': 'Log Target MAPE', 'four_level_gb': 'Log Target MAPE', 'earned_relative_total_value': '32%'},
            {'interval': '[μ + 3σ : μ + 5σ]', 'two_level_gb': '-', 'three_level_gb': 'Log Target MAPE', 'four_level_gb': 'Log Target MAPE', 'earned_relative_total_value': '22%'},
            {'interval': '[μ + 5σ : μ + 12σ]', 'two_level_gb': '-', 'three_level_gb': '-', 'four_level_gb': 'Log Target MAPE', 'earned_relative_total_value': '18%'},
        ],
        style_table={'margin-left': '50px', "width": "1000px"},
        style_cell={'textAlign': 'center','color': 'white', 'backgroundColor': '#000'},
    ),
    
    
    
    
    
    
    html.Div(''' 
             Признаки которые можно было добавить:
             Криминальность района
             Доход района - Средний доход жителей района, в котором находится жилье. Это может влиять на уровень цен на аренду
             Индекс стоимости жизни - Индекс, отражающий общую стоимость жизни в конкретном городе или районе. Это может влиять на уровень цен
             Уровень безработицы - Процент безработных в районе может быть связан с общим спросом на аренду
             Инфляция - Уровень инфляции может влиять на уровень цен на жилье
             Стоимость коммунальных услуг в Италии
             Ставка привлечения капитала - Процентная ставка или ставка привлечения капитала может влиять на инвестиции в недвижимость
             Ставка налога на недвижимость - Сведения о ставке налога на недвижимость в конкретном районе
             Инвестиционные показатели - Например, количество новых строительств и инвестиций в недвижимость3 
             Показатели финансовой стабильности - Например, рейтинги кредитоспособности района или города
             ''', style={'height': '200px', "margin-top": "30px"}),  
], 
    style={"backgroundColor": dark_theme["main-background"]}, 
)

if __name__ == '__main__':
    app.run_server(debug=True)
