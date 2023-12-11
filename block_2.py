import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template


from app import densitymapbox_price, df_geo_city, histogram_distance_to_centre_city, histogram_cat_distance


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



def create_explore_data_analysis_block_2():
    return html.Div([
    html.H3("Data Engineering", style={'margin': '60px'}),
    
    
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
    ])