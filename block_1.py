import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template

from app import histogram_sourse, histogram_host_response_time, histogram_host_response_rate, histogram_host_acceptance_rate, histogram_host_is_superhost, histogram_host_listings_count, histogram_host_total_listings_count, histogram_host_verifications, histogram_accommodates, histogram_number_of_reviews, histogram_host_has_profile_pic


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



def create_explore_data_analysis_block():
    return html.Div([
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
    ])