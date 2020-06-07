# -*- coding: utf-8 -*-
import collections
import dash
import dill
import json
from more_itertools import divide
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd

import plotly.graph_objects as go

import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output


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
    my_path = 'models'
    available_methods_unsorted = [f.split('__')[0] for f in listdir(my_path) if isfile(join(my_path, f)) & ('evo_training' in f)]
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

fake_parameters_episode_table = pd.DataFrame(columns=[
    'nmax_steps'
])


available_methods = get_available_methods()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(

    children=[

        # Hidden div inside the app that stores the intermediate value
        html.Div(id='list_evo_real_episode', style={'display': 'none'})
        , html.Div(id='list_evo_training', style={'display': 'none'})
        , html.Div(id='list_evo_one_episode', style={'display': 'none'})

        , html.H1(
            children='DOGGO - Analysis'
        )

        , html.Hr()

        , html.Div([
            dcc.Dropdown(
                id='list_method_id',
                options=[{'label': i, 'value': i} for i in available_methods],
                value='',
                multi=True
                )
            ],
            style={'width': '100%', 'display': 'inline-block'})

        , html.Div([
            dash_table.DataTable(
                id='parameters_table',
                columns=[{"name": i, "id": i} for i in fake_parameters_table.columns],
                page_current=0,
                page_size=5,
                page_action='custom')])

        , html.Hr()

        , html.H3(
            children='Real episodes'
        )

        , html.Div([
            dash_table.DataTable(
                id='parameters_episode_table',
                columns=[{"name": i, "id": i} for i in fake_parameters_episode_table.columns],
                page_current=0,
                page_size=5,
                page_action='custom')
        ], style={'display': 'inline-block', 'width': '100%'})

        , html.Div([
            dcc.Graph(id='rl_evo_real_reward')
        ], style={'display': 'inline-block', 'width': '49%'})

        , html.Div([
            dcc.Graph(id='rl_evo_real_steps')
        ], style={'display': 'inline-block', 'width': '49%'})

        , html.Hr()

        , html.H3(
            children='During training'
        )

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

        , html.Hr()

        , html.H3(
            children='DOGGO specific'
        )
        , html.Div([
            dcc.Graph(id='specific_evo_real_happiness')
        ], style={'display': 'inline-block', 'width': '33%'})
        , html.Div([
            dcc.Graph(id='specific_evo_training_happiness')
        ], style={'display': 'inline-block', 'width': '33%'})
        , html.Div([
            dcc.Graph(id='specific_evo_real_n_actions')
        ], style={'display': 'inline-block', 'width': '33%'})

        , html.Hr()

        , html.H4(
            children='DOGGO specific - characteristics'
        )

        , dcc.RadioItems(id='i_method_id')

        , html.Div([
            dcc.Graph(id='specific_evo_real_characteristics')
        ], style={'display': 'inline-block', 'width': '49%'})

        , html.Div([
            dcc.Graph(id='specific_evo_real_cause_of_death')
        ], style={'display': 'inline-block', 'width': '49%'})

        , html.Hr()

        , html.H4(
            children='DOGGO specific - one episode'
        )

        , html.Div([
            dcc.Graph(id='specific_evo_real_one_episode')
        ], style={'display': 'inline-block', 'width': '100%'})

    ]
)


@app.callback(
    Output('parameters_table', 'data'),
    [Input('list_method_id', 'value')])
def update_display_parameters(list_method_id):

    parameters_table = pd.DataFrame()

    for method_id in list_method_id:
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
    Output('list_evo_real_episode', 'children'),
    [Input('list_method_id', 'value')])
def fetch_data_evo_real_episode(list_method_id):

    list_evolution_real_episode_reduced = dict()

    for method_id in list_method_id:

        with open("models/{}__evolution_real_episode.pkl".format(method_id), "rb") as input_file:
            evolution_real_episode = dill.load(input_file)

        evolution_real_episode_reduced = dict()
        for i_key in evolution_real_episode.keys():
            if i_key == 'cause_of_death':
                evolution_real_episode_reduced[i_key] = evolution_real_episode[i_key]
            else:
                evolution_real_episode_reduced[i_key] = get_avg_n_points(evolution_real_episode[i_key], n_points=100)

        list_evolution_real_episode_reduced[method_id] = evolution_real_episode_reduced

    return list_evolution_real_episode_reduced


@app.callback(
    Output('list_evo_training', 'children'),
    [Input('list_method_id', 'value')])
def fetch_data_evo_training(list_method_id):

    list_evo_training_reduced = dict()

    for method_id in list_method_id:

        with open("models/{}__evo_training.pkl".format(method_id), "rb") as input_file:
            evo_training = dill.load(input_file)
        evo_training = flatten(evo_training)

        evo_training_reduced = dict()
        for i_key in evo_training.keys():
            if i_key in ['evo_avg_reward_per_step', 'evo_min_reward_per_step', 'evo_max_reward_per_step', 'evo_n_steps',
                         'evo_avg_happiness', 'evo_min_happiness', 'evo_max_happiness', 'checking__evo_epsilon',
                         'checking__evo_alpha']:
                evo_training_reduced[i_key] = get_avg_n_points(evo_training[i_key], n_points=100)

        list_evo_training_reduced[method_id] = evo_training_reduced

    return list_evo_training_reduced


@app.callback(
    Output('list_evo_one_episode', 'children'),
    [Input('list_method_id', 'value')])
def fetch_data_evo_one_episode(list_method_id):

    list_evo_one_episode_reduced = dict()

    for method_id in list_method_id:

        with open("models/{}__final__evo_episode.pkl".format(method_id), "rb") as input_file:
            evo_one_episode = dill.load(input_file)

        list_evo_one_episode_reduced[method_id] = evo_one_episode

    return list_evo_one_episode_reduced


@app.callback(
    Output('parameters_episode_table', 'data'),
    [Input('list_method_id', 'value')])
def update_display_parameters_episode(list_method_id):

    parameters_episode_table = pd.DataFrame()

    params_episode = json.loads(open('src/models/run_one_episode.json').read())

    parameters_episode_table = parameters_episode_table.append(params_episode, ignore_index=True)

    parameters_episode_table = parameters_episode_table[[
        'nmax_steps'
    ]]

    return parameters_episode_table.to_dict('records')


@app.callback(
    Output('rl_evo_real_reward', 'figure'),
    [Input('list_evo_real_episode', 'children')])
def update_graph_rl_evo_real_reward(list_evo_real_episode):

    list_data = []

    for i_method_id in list_evo_real_episode.keys():

        list_data.append({
                'x': list(range(len(list_evo_real_episode[i_method_id]['avg_reward']))),
                'y': list_evo_real_episode[i_method_id]['avg_reward'],
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
                'title': 'Reward',
                'range': [-0.5, 1.5]
            },
            title='Evolution of average reward per episode'
        )
    }


@app.callback(
    Output('rl_evo_real_steps', 'figure'),
    [Input('list_evo_real_episode', 'children')])
def update_graph_rl_evo_real_steps(list_evo_real_episode):

    list_data = []

    for i_method_id in list_evo_real_episode.keys():

        list_data.append({
                'x': list(range(len(list_evo_real_episode[i_method_id]['n_steps']))),
                'y': list_evo_real_episode[i_method_id]['n_steps'],
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
                'title': '#Steps'
            },
            title='Evolution of the number of steps per episode'
        )
    }


@app.callback(
    Output('rl_evo_training_reward', 'figure'),
    [Input('list_evo_training', 'children')])
def update_graph_rl_evo_training_reward(list_evo_training):

    list_data = []

    for i_method_id in list_evo_training.keys():

        list_data.append(
            {
                'x': list(range(len(list_evo_training[i_method_id]['evo_avg_reward_per_step']))),
                'y': list_evo_training[i_method_id]['evo_avg_reward_per_step'],
                'type': 'line',
                'name': i_method_id  # + ' - average'
            })
        # list_data.append({
        #         'x': list(range(len(list_evo_training[i_method_id]['evo_min_reward_per_step']))),
        #         'y': list_evo_training[i_method_id]['evo_min_reward_per_step'],
        #         'type': 'line',
        #         'name': i_method_id + ' - min',
        #         'line': {'dash': 'dot', 'color': 'lightcoral'}
        #     })
        # list_data.append({
        #         'x': list(range(len(list_evo_training[i_method_id]['evo_max_reward_per_step']))),
        #         'y': list_evo_training[i_method_id]['evo_max_reward_per_step'],
        #         'type': 'line',
        #         'name': i_method_id + ' - max',
        #         'line': {'dash': 'dot', 'color': 'lightgreen'}
        #     })

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
    [Input('list_evo_training', 'children')])
def update_graph_rl_evo_training_steps(list_evo_training):

    list_data = []

    for i_method_id in list_evo_training.keys():

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
                'title': '#Steps'
            },
            title='Steps',
            legend={'orientation': 'h', 'y': -0.5}
        )
    }


@app.callback(
    Output('rl_evo_training_epsilon', 'figure'),
    [Input('list_evo_training', 'children')])
def update_graph_rl_evo_training_epsilon(list_evo_training):

    list_data = []

    for i_method_id in list_evo_training.keys():

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
                'title': 'Epsilon'
            },
            title='Epsilon',
            legend={'orientation': 'h', 'y': -0.5}
        )
    }


@app.callback(
    Output('rl_evo_training_alpha', 'figure'),
    [Input('list_evo_training', 'children')])
def update_graph_rl_evo_training_alpha(list_evo_training):

    list_data = []

    for i_method_id in list_evo_training.keys():

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
                'title': 'Alpha'
            },
            title='Alpha',
            legend={'orientation': 'h', 'y': -0.5}
        )
    }


@app.callback(
    Output('specific_evo_real_happiness', 'figure'),
    [Input('list_evo_real_episode', 'children')])
def update_graph_specific_evo_real_happiness(list_evo_real_episode):

    list_data = []

    for i_method_id in list_evo_real_episode.keys():

        list_data.append({
                'x': list(range(len(list_evo_real_episode[i_method_id]['avg_happiness']))),
                'y': list_evo_real_episode[i_method_id]['avg_happiness'],
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
                'title': 'Happiness',
                'range': [0, 1]
            },
            title='Happiness (real)'
        )
    }


@app.callback(
    Output('specific_evo_training_happiness', 'figure'),
    [Input('list_evo_training', 'children')])
def update_graph_training_evo_happiness(list_evo_training):

    list_data = []

    for i_method_id in list_evo_training.keys():

        list_data.append({
            'x': list(range(len(list_evo_training[i_method_id]['evo_avg_happiness']))),
            'y': list_evo_training[i_method_id]['evo_avg_happiness'],
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
                'title': 'Happiness',
                'range': [0, 1]
            },
            title='Happiness (training)'
        )
    }


@app.callback(
    Output('specific_evo_real_n_actions', 'figure'),
    [Input('list_evo_real_episode', 'children')])
def update_graph_specific_evo_real_n_actions(list_evo_real_episode):

    list_data = []

    for i_method_id in list_evo_real_episode.keys():

        avg_action_per_step = [x / y for x, y in zip(
            list_evo_real_episode[i_method_id]['n_actions'], list_evo_real_episode[i_method_id]['n_steps'])]

        list_data.append({
                'x': list(range(len(avg_action_per_step))),
                'y': avg_action_per_step,
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
                'title': '#Actions / #Steps'
            },
            title='#Actions / #Steps'
        )
    }


@app.callback(
    Output('i_method_id', 'options'),
    [Input('list_method_id', 'value')])
def set_method_id_options(list_method_id):
    return [{'label': i, 'value': i} for i in list_method_id]


@app.callback(
    Output('i_method_id', 'value'),
    [Input('i_method_id', 'options')])
def set_method_id_value(i_method_id):
    return i_method_id[0]['value']


@app.callback(
    Output('specific_evo_real_characteristics', 'figure'),
    [Input('list_evo_real_episode', 'children'),
     Input('i_method_id', 'value')])
def update_graph_specific_evo_real_characteristics(list_evo_real_episode, i_method_id):

    list_data = list()

    list_data.append({
            'x': list(range(len(list_evo_real_episode[i_method_id]['avg_food']))),
            'y': list_evo_real_episode[i_method_id]['avg_food'],
            'type': 'line',
            'name': 'food',
            'line': {'color': 'lightgreen'}
        })
    list_data.append({
        'x': list(range(len(list_evo_real_episode[i_method_id]['avg_inv_fat']))),
        'y': list_evo_real_episode[i_method_id]['avg_inv_fat'],
        'type': 'line',
        'name': 'inverse fat',
        'line': {'color': 'lightcoral'}
    })
    list_data.append({
        'x': list(range(len(list_evo_real_episode[i_method_id]['avg_affection']))),
        'y': list_evo_real_episode[i_method_id]['avg_affection'],
        'type': 'line',
        'name': 'affection',
        'line': {'color': 'lightblue'}
    })

    return {
        'data': list_data,
        'layout': dict(
            xaxis={
                'title': 'Episodes'
            },
            yaxis={
                'title': 'Characteristics',
                'range': [0, 1]
            },
            title='Food/Fat/Affection'
        )
    }


@app.callback(
    Output('specific_evo_real_cause_of_death', 'figure'),
    [Input('list_evo_real_episode', 'children'),
     Input('i_method_id', 'value')])
def update_graph_specific_evo_real_cause_of_death(list_evo_real_episode, i_method_id):

    n_points = 100

    cause_of_death_divided = [list(l) for l in divide(n_points, list_evo_real_episode[i_method_id]['cause_of_death'])]

    cause_of_death_pct = {
        'food': [],
        'fat': [],
        'affection': [],
        'no_death': []
    }

    for i_list in cause_of_death_divided:
        cause_of_death_pct['food'].append(int(np.floor(len([x for x in i_list if x == 'food']) / len(i_list) * 100)))
        cause_of_death_pct['fat'].append(int(np.floor(len([x for x in i_list if x == 'fat']) / len(i_list) * 100)))
        cause_of_death_pct['affection'].append(
            int(np.floor(len([x for x in i_list if x == 'affection']) / len(i_list) * 100)))
        cause_of_death_pct['no_death'].append(
            int(np.floor(len([x for x in i_list if x == 'no_death']) / len(i_list) * 100)))

    x = list(range(n_points))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x, y=cause_of_death_pct['no_death'],
        name='no_death', width=0.95, marker_color=['seashell', ]*n_points))
    fig.add_trace(go.Bar(
        x=x, y=cause_of_death_pct['affection'],
        name='affection', width=0.95, marker_color=['lightblue', ]*n_points))
    fig.add_trace(go.Bar(
        x=x, y=cause_of_death_pct['fat'],
        name='fat', width=0.95, marker_color=['lightcoral', ]*n_points))
    fig.add_trace(go.Bar(
        x=x, y=cause_of_death_pct['food'],
        name='food', width=0.95, marker_color=['lightgreen', ]*n_points))

    fig.update_layout(
        barmode='stack',
        xaxis={'categoryorder': 'category ascending'},
        yaxis=dict(range=[0, 100])
    )

    return fig


@app.callback(
    Output('specific_evo_real_one_episode', 'figure'),
    [Input('list_evo_one_episode', 'children'),
     Input('i_method_id', 'value')])
def update_graph_specific_evo_one_episode(list_evo_one_episode, i_method_id):

    # evo_one_episode = list_evo_one_episode[i_method_id]

    x = np.arange(len(list_evo_one_episode[i_method_id]['happiness']))

    # Create traces
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=list_evo_one_episode[i_method_id]['happiness'],
        legendgroup="happiness",
        mode='lines',
        name='happiness',
        marker_color='black', line={'width': 2}
    ))

    fig.add_trace(go.Scatter(
        x=x, y=list_evo_one_episode[i_method_id]['food'],
        legendgroup="characteristics",
        mode='lines',
        name='food',
        marker_color='lightgreen', line={'width': 1}
    ))

    fig.add_trace(go.Scatter(
        x=x, y=list_evo_one_episode[i_method_id]['inv_fat'],
        legendgroup="characteristics",
        mode='lines',
        name='inv. fat',
        marker_color='lightcoral', line={'width': 1}
    ))

    fig.add_trace(go.Scatter(
        x=x, y=list_evo_one_episode[i_method_id]['affection'],
        legendgroup="characteristics",
        mode='lines',
        name='affection',
        marker_color='lightblue', line={'width': 1}
    ))

    action_dict = {
        0: ['NO ACTION', 'white'],
        1: ['WALKING', 'indigo'],
        2: ['EATING', 'hotpink'],
        3: ['PLAYING', 'darkorange']
    }

    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, 1],
        legendgroup="taken",
        mode='lines',
        name=action_dict[1][0] + ' - taken',
        marker_color=action_dict[1][1],
        line={'width': 1, 'dash': 'solid'}
    ))

    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, 1],
        legendgroup="non-taken",
        mode='lines',
        name=action_dict[1][0] + ' - non-taken',
        marker_color=action_dict[1][1],
        line={'width': 1, 'dash': 'dot'}
    ))

    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, 1],
        legendgroup="taken",
        mode='lines',
        name=action_dict[2][0] + ' - taken',
        marker_color=action_dict[2][1],
        line={'width': 1, 'dash': 'solid'}
    ))

    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, 1],
        legendgroup="non-taken",
        mode='lines',
        name=action_dict[2][0] + ' - non-taken',
        marker_color=action_dict[2][1],
        line={'width': 1, 'dash': 'dot'}
    ))

    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, 1],
        legendgroup="taken",
        mode='lines',
        name=action_dict[3][0] + ' - taken',
        marker_color=action_dict[3][1],
        line={'width': 1, 'dash': 'solid'}
    ))

    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, 1],
        legendgroup="non-taken",
        mode='lines',
        name=action_dict[3][0] + ' - non-taken',
        marker_color=action_dict[3][1],
        line={'width': 1, 'dash': 'dot'}
    ))

    for i in range(len(list_evo_one_episode[i_method_id]['action_taken'])):

        if list_evo_one_episode[i_method_id]['action_taken'][i] != 0:
            fig.add_trace(go.Scatter(
                x=[i, i], y=[0, 1],
                mode='lines',
                name=action_dict[list_evo_one_episode[i_method_id]['action_taken'][i]][0],
                marker_color=action_dict[list_evo_one_episode[i_method_id]['action_taken'][i]][1],
                line={'width': 0.5, 'dash': 'solid'},
                showlegend=False
            ))
        elif list_evo_one_episode[i_method_id]['action'][i] != 0:
            fig.add_trace(go.Scatter(
                x=[i, i], y=[0, 1],
                mode='lines',
                name=action_dict[list_evo_one_episode[i_method_id]['action'][i]][0],
                marker_color=action_dict[list_evo_one_episode[i_method_id]['action'][i]][1],
                line={'width': 0.5, 'dash': 'dot'},
                showlegend=False
            ))

    fig.update_layout(
        title={
            'text': "One episode",
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Steps",
        yaxis_title="State",
        legend={'itemsizing': 'constant'}
    )

    return fig


if __name__ == '__main__':
    app.run_server(
        debug=False,
        port=8051)
