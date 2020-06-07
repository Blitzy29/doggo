# -*- coding: utf-8 -*-
import collections
import dash
import dill
import json
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd

import plotly.graph_objects as go

import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output


# python3 src/visualization/app_current

def flatten(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def running_mean(x, n_points):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n_points:] - cumsum[:-n_points]) / float(n_points)


def get_avg_n_points(list_points, n_points):
    n_moving_points = int(np.ceil(len(list_points) / n_points))
    y_all = running_mean(list_points, n_moving_points)
    y = [list_points[0]] + [j for i, j in enumerate(y_all) if i % n_moving_points == 0] + [y_all[-1]]
    return y


def get_available_methods():
    my_path = 'data/log/pid'
    available_methods_unsorted = [f.split('.')[0] for f in listdir(my_path) if isfile(join(my_path, f)) & ('.pid' in f)]
    print(available_methods_unsorted)
    if len(available_methods_unsorted) == 0:
        return ['']
    return np.sort(available_methods_unsorted)


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


fake_parameters_table = pd.DataFrame(columns=[
    'method', 'method_id',
    'n_episodes', 'nmax_steps', 'gamma', 'start_at_random',
    'epsilon__init_epsilon', 'epsilon__decay_epsilon', 'epsilon__min_epsilon',
    'alpha__init_alpha', 'alpha__decay_alpha', 'alpha__min_alpha',
    'init_Q_type', 'KLdiv_convergence'
])


available_methods = get_available_methods()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(

    children=[

        # Hidden div inside the app that stores the intermediate value
        html.Div(id='list_evo_training', style={'display': 'none'})

        , html.H1(
            children='DOGGO - Currently running'
        )

        , html.Hr()

        , html.Div([
            dash_table.DataTable(
                id='parameters_table',
                columns=[{"name": i, "id": i} for i in fake_parameters_table.columns],
                page_current=0,
                page_size=5,
                page_action='custom')])

        , dcc.Markdown(id='parameters')

        , html.Div([
            dcc.Graph(id='progress')
        ], style={'display': 'inline-block', 'width': '100%'})

        , html.Hr()

        , html.Div([
            dcc.Dropdown(
                id='i_method_id',
                options=[{'label': i, 'value': i} for i in available_methods],
                value=''
            )
        ],
            style={'width': '100%', 'display': 'inline-block'})

        , html.Hr()

        , html.Div([
            dcc.Graph(id='rl_evo_training_reward')
        ], style={'display': 'inline-block', 'width': '49%'})
        , html.Div([
            dcc.Graph(id='rl_evo_training_steps')
        ], style={'display': 'inline-block', 'width': '49%'})

        , html.Div([
            dcc.Graph(id='rl_evo_training_epsilon')
        ], style={'display': 'inline-block', 'width': '49%'})
        , html.Div([
            dcc.Graph(id='rl_evo_training_alpha')
        ], style={'display': 'inline-block', 'width': '49%'})


    ]
)

@app.callback(
    Output('list_evo_training', 'children'),
    [Input('i_method_id', 'value')])
def fetch_data_evo_training(i_method_id):

    list_evo_training = dict()

    for method_id in available_methods:

        with open("data/interim/{}__evo_training.pkl".format(method_id), "rb") as input_file:
            evo_training = dill.load(input_file)
        evo_training = flatten(evo_training)

        if 'sarsa' in method_id:
            params_method = json.loads(open('src/models/value_based/sarsa/config/{}.json'.format(method_id)).read())
        elif 'doubleqlearning' in method_id:
            params_method = json.loads(open('src/models/value_based/doubleqlearning/config/{}.json'.format(method_id)).read())
        elif 'qlearning' in method_id:
            params_method = json.loads(open('src/models/value_based/qlearning/config/{}.json'.format(method_id)).read())
        elif 'monte_carlo' in method_id:
            params_method = json.loads(open('src/models/value_based/monte_carlo/config/{}.json'.format(method_id)).read())
        evo_training['n_episodes'] = params_method['n_episodes']
        evo_training['nmax_steps'] = params_method['nmax_steps']

        list_evo_training[method_id] = evo_training

    return list_evo_training


@app.callback(
    Output('parameters_table', 'data'),
    [Input('list_evo_training', 'children')])
def update_display_parameters(list_evo_training):

    parameters_table = pd.DataFrame()

    for method_id in list_evo_training.keys():
        if 'sarsa' in method_id:
            params_method = json.loads(open('src/models/value_based/sarsa/config/{}.json'.format(method_id)).read())
        elif 'doubleqlearning' in method_id:
            params_method = json.loads(open('src/models/value_based/doubleqlearning/config/{}.json'.format(method_id)).read())
        elif 'qlearning' in method_id:
            params_method = json.loads(open('src/models/value_based/qlearning/config/{}.json'.format(method_id)).read())
        elif 'monte_carlo' in method_id:
            params_method = json.loads(open('src/models/value_based/monte_carlo/config/{}.json'.format(method_id)).read())
        params_method['method_id'] = method_id

        parameters_table = parameters_table.append(flatten(params_method), ignore_index=True)

    parameters_table = parameters_table[[
        'method', 'method_id',
        'n_episodes', 'nmax_steps', 'gamma', 'start_at_random',
        'epsilon__init_epsilon', 'epsilon__decay_epsilon', 'epsilon__min_epsilon',
        'alpha__init_alpha', 'alpha__decay_alpha', 'alpha__min_alpha',
        'init_Q_type', 'KLdiv_convergence'
    ]]

    return parameters_table.to_dict('records')


@app.callback(
    Output('progress', 'figure'),
    [Input('list_evo_training', 'children')])
def update_graph_progress(list_evo_training):

    max_value_overall = 0

    fig = go.Figure()

    for i_method_id in list_evo_training.keys():

        progress_value = len(list_evo_training[i_method_id]['evo_avg_reward_per_step'])
        max_value = list_evo_training[i_method_id]['n_episodes']

        if progress_value / max_value < 0.33:
            color_progress = 'lightcoral'
        elif progress_value / max_value < 0.66:
            color_progress = 'lightgoldenrodyellow'
        elif (progress_value+1) / max_value < 1:
            color_progress = 'lightgreen'
        else:
            color_progress = 'green'

        fig.add_trace(go.Bar(
            x=[progress_value],
            y=[i_method_id],
            name='progress',
            orientation='h',
            marker=dict(
                color=color_progress
            )
        ))

        max_value_overall = max(max_value_overall, max_value)

    fig.update_layout(
        xaxis=dict(range=[0, max_value_overall]),
        showlegend=False,
        height=300
    )

    return fig


@app.callback(
    Output('rl_evo_training_reward', 'figure'),
    [Input('list_evo_training', 'children'),
     Input('i_method_id', 'value')])
def update_graph_rl_evo_training_reward(list_evo_training, i_method_id):

    list_data = []

    list_data.append(
        {
            'x': list(range(len(list_evo_training[i_method_id]['evo_avg_reward_per_step']))),
            'y': list_evo_training[i_method_id]['evo_avg_reward_per_step'],
            'type': 'line',
            'name': i_method_id  # + ' - average'
        })
    list_data.append({
            'x': list(range(len(list_evo_training[i_method_id]['evo_min_reward_per_step']))),
            'y': list_evo_training[i_method_id]['evo_min_reward_per_step'],
            'type': 'line',
            'name': i_method_id + ' - min',
            'line': {'dash': 'dot', 'color': 'lightcoral'}
        })
    list_data.append({
            'x': list(range(len(list_evo_training[i_method_id]['evo_max_reward_per_step']))),
            'y': list_evo_training[i_method_id]['evo_max_reward_per_step'],
            'type': 'line',
            'name': i_method_id + ' - max',
            'line': {'dash': 'dot', 'color': 'lightgreen'}
        })

    return {
        'data': list_data,
        'layout': dict(
            xaxis={
                'title': 'Episodes (pct)'
            },
            yaxis={
                'title': 'Reward',
                'range': [-0.5, 1.5]
            },
            title='Reward',
            legend={'orientation': 'h', 'y': -0.5}
        )
    }


@app.callback(
    Output('rl_evo_training_steps', 'figure'),
    [Input('list_evo_training', 'children'),
     Input('i_method_id', 'value')])
def update_graph_rl_evo_training_steps(list_evo_training, i_method_id):

    list_data = []

    list_data.append({
        'x': list(range(len(list_evo_training[i_method_id]['evo_n_steps']))),
        'y': list_evo_training[i_method_id]['evo_n_steps'],
        'type': 'line',
        'name': i_method_id
        })

    return {
        'data': list_data,
        'layout': dict(
            xaxis={
                'title': 'Episodes (pct)'
            },
            yaxis={
                'title': '#Steps',
                'range': [0, list_evo_training[i_method_id]['nmax_steps']]
            },
            title='Steps',
            legend={'orientation': 'h', 'y': -0.5}
        )
    }


@app.callback(
    Output('rl_evo_training_epsilon', 'figure'),
    [Input('list_evo_training', 'children'),
     Input('i_method_id', 'value')])
def update_graph_rl_evo_training_epsilon(list_evo_training, i_method_id):

    list_data = []

    list_data.append({
        'x': list(range(len(list_evo_training[i_method_id]['checking__evo_epsilon']))),
        'y': list_evo_training[i_method_id]['checking__evo_epsilon'],
        'type': 'line',
        'name': i_method_id
        })

    return {
        'data': list_data,
        'layout': dict(
            xaxis={
                'title': 'Steps (pct)'
            },
            yaxis={
                'title': 'Epsilon',
                'range': [0, 1]
            },
            title='Epsilon',
            legend={'orientation': 'h', 'y': -0.5}
        )
    }


@app.callback(
    Output('rl_evo_training_alpha', 'figure'),
    [Input('list_evo_training', 'children'),
     Input('i_method_id', 'value')])
def update_graph_rl_evo_training_alpha(list_evo_training, i_method_id):

    list_data = []

    list_data.append({
        'x': list(range(len(list_evo_training[i_method_id]['checking__evo_alpha']))),
        'y': list_evo_training[i_method_id]['checking__evo_alpha'],
        'type': 'line',
        'name': i_method_id
        })

    return {
        'data': list_data,
        'layout': dict(
            xaxis={
                'title': 'Steps (pct)'
            },
            yaxis={
                'title': 'Alpha',
                'range': [0, 1]
            },
            title='Alpha',
            legend={'orientation': 'h', 'y': -0.5}
        )
    }


if __name__ == '__main__':
    app.run_server(
        debug=False,
        port=8052)

