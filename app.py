import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template

file_id ="1XmvSJLpBzT8rjAOB2C6jLB1O_7W3ruyu"
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




app.layout = dbc.Container([
    html.H2("Predicting Real Estate Prices from the Airbnb Portal"),
    html.H3("Content"),
    
    dcc.Markdown('''
    - [Explore Data Analysis](#explore-data-analysis)
    - [Data Preparation](#data-preparation)
    - [Feature Engineering](#feature-engineering)
    '''),
    
    html.H3("Explore Data Analysis", style={'margin': '50px'}),

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
    
    # dcc.Graph(
    #     id='heatmap',
    #     figure=heatmap_fig,
    #     style={'border': '2px solid black', "margin-bottom":"50px"}
    # ),
    
    dcc.Markdown('''
        ### Target - Price
        Заметим, что денежные значения у целового признака представлены в разных валютах '$' '€' '₽'.
        
        В наборе данных присутсвует признак "calendar_last_scraped" - "последние обновление данных". Эти даты я собираюсь использовать для конвертации валют по корректному курсу.
        
        Возьмем курсы валют с официального портала Банк России и конвертируем валюты.
    ''', style={'margin-left': '50px','fontSize': '18px'}),
    
    
    # dcc.Graph(
    #     id='price_fig',
    #     figure=price_fig,
    #     style={'border': '2px solid black', "margin-bottom":"20px"}
    # ),
    
    
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
    
    
    
    # generate_hypothesis_block("Biserial", "sourse", "price_usd", 0.01143, 0.00056, 0.05, True),
    # dcc.Graph(
    #     id='histogram_source',
    #     figure=histogram_sourse,
    #     style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    # ),
    
    # generate_hypothesis_block("Biserial", "host_response_time", "price_usd", 0.00564, 0.08907, 0.05, False),
    # dcc.Graph(
    #     id='histogram_host_response_time',
    #     figure=histogram_host_response_time,
    #     style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    # ),
    
    # generate_hypothesis_block("Pearson", "host_response_rate", "price_usd", -0.02782, 0.0, 0.05, True),
    # dcc.Graph(
    #     id='histogram_host_response_rate',
    #     figure=histogram_host_response_rate,
    #     style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    # ), 
    
    # generate_hypothesis_block("Pearson", "host_acceptance_rate", "price_usd", 0.05246, 0.0, 0.05, True),
    # dcc.Graph(
    #     id='histogram_host_acceptance_rate',
    #     figure=histogram_host_acceptance_rate,
    #     style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    # ), 
    
    # generate_hypothesis_block("Biserial", "host_is_superhost", "price_usd", 0.01452, 1e-05, 0.05, True),
    # dcc.Graph(
    #     id='histogram_host_is_superhost',
    #     figure=histogram_host_is_superhost,
    #     style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    # ), 

    # generate_hypothesis_block("Pearson", "host_listings_count", "price_usd", 0.11413, 0.0, 0.05, True),
    # dcc.Graph(
    #     id='histogram_host_listings_count',
    #     figure=histogram_host_listings_count,
    #     style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    # ),
    # generate_hypothesis_block("Pearson", "host_total_listings_count", "price_usd", 0.09509, 0.0, 0.05, True),
    # dcc.Graph(
    #     id='histogram_host_total_listings_count',
    #     figure=histogram_host_total_listings_count,
    #     style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    # ), 
    # generate_hypothesis_block("Biserial", "host_verifications", "price_usd", 0.03171, 0.00056, 0.05, True),
    # dcc.Graph(
    #     id='histogram_host_verifications',
    #     figure=histogram_host_verifications,
    #     style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    # ), 
    # generate_hypothesis_block("Pearson", "accommodates", "price_usd", 0.35318, 0.00056, 0.05, True),
    # dcc.Graph(
    #     id='histogram_accommodates',
    #     figure=histogram_accommodates,
    #     style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    # ),
    # generate_hypothesis_block("Pearson", "number_of_reviews", "price_usd", -0.04753, 0.00056, 0.05, True),
    # dcc.Graph(
    #     id='histogram_number_of_reviews',
    #     figure=histogram_number_of_reviews,
    #     style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    # ),
    generate_hypothesis_block("Pearson", "host_has_profile_pic", "price_usd", 0.01902, 0.00056, 0.05, True),
    dcc.Graph(
        id='histogram_host_has_profile_pic',
        figure=histogram_host_has_profile_pic,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ),
    
    
    
    html.H3("Data Engineering", style={'margin': '60px'}),
    
    
    # html.Div([
    #     dcc.Markdown('''
    #         Используем `latitude` и `longitude` для представления географических координат недвижимости в разных городах Италии.
    #     '''),
    # ], style={'fontSize': '18px', 'margin-left': '50px'}),
    
    dcc.Graph(
        id='densitymapbox_price',
        figure=densitymapbox_price,
    ),
    
    
    html.Div([
        dcc.Markdown('''
            Извлекая информацию из общедоступного интернет-ресурса, я получил географические координаты (долготу и широту) городов Италии, упомянутых в предоставленных данных об объектах недвижимости. 
            Далее я вычислил расстояние каждого объекта до центра города, в котором он расположен, и создал два новых признака для анализа.

            Первый признак, `distance_to_centre_city`, является непрерывным и отражает фактическое расстояние в километрах от объекта недвижимости до центра города. 

            Второй признак, `cat_distance`, является категориальным и представляет расстояние в определенных диапазонах. 
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
    generate_hypothesis_block("Pearson", "cat_distance", "price_usd", -0.02601, 0.0, 0.05, True),
    dcc.Graph(
        id='histogram_cat_distance',
        figure=histogram_cat_distance,
        style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-top': '60px'}
    ),
    
    
    html.Div("", style={
        'height': '200px'
    }),  
], 
                      style={"backgroundColor": dark_theme["main-background"]}, 
)

if __name__ == '__main__':
    app.run_server(debug=True)
