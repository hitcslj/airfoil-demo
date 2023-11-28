import random
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np
# import torch



def _graph_resize(graph):
    graph = graph.reshape((len(graph)//2, 2))
    return graph


def pca(app, model, dataset, latent_options, pre_process=None, prefix=""):
    prefix += '-pca-'

    header = dbc.Row(
        [html.Div(html.H5("Airfoil Parameter Visualization Demo"), className="col-md-6")])

    dataset_unscaled = pre_process.inverse_transform(dataset)
    data_pca = model.transform(dataset)

    # Left input Image Display Area
    input_div = dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Span(children="", id=prefix +
                          'input-content-id', className='p-2 d-grid gap-2'),
                html.Span(children="", id=prefix +
                          'input-content-id-hidden', className='d-none'),

                dbc.Button('sample', color="primary", id=prefix +
                           'sample-input', className="mr-1 float-right", n_clicks=0),
            ], 
            className="d-grid gap-2 d-md-flex justify-content-between", #d-grid gap-2 d-md-flex justify-content-between
            style={'height': '40px'})
        ]),
        dbc.CardBody(
            [
                dcc.Graph(
                    # style={ 'width': 200},
                    id=prefix + 'input-content',
                ),

            ], 
            className="d-flex justify-content-center",
            style={'height': '480px'}
        ),
    ]
    )

    # Middle Slider Area
    latent_size = latent_options['n']
    latent_space = []
    # just used to fill initial space in the html
    init_hidden_space = [str(latent_options['min'])
                         for _ in range(latent_size)]

    for _ in range(latent_size):
        id = prefix + 'latent-slider-' + str(_)
        latent_space.append(dcc.Slider(min=latent_options['min'],
                                       max=latent_options['max'],
                                       marks={latent_options['min']: f"{latent_options['min']}",
                                              0: '0',
                                              latent_options['max']: f"{latent_options['max']}"},
                                       step=latent_options['step'],
                                       updatemode='drag',
                                       id=id,
                                       value=latent_options['min'],
                                       tooltip={"placement": "bottom",
                                                "always_visible": False},
                                       className="mt-3 mb-3"))

    # Reference Range
    md_ref_str = "```\n"
    recommded_range = [[-13, 30], [-30, 30], [-8, 30], [-30, 30], [-12, 12], [-10, 5]]
    for _ in range(latent_size):
        if _ < len(recommded_range):
            md_ref_str += f"Param {_+1}: {recommded_range[_]}\n"
        else:
            md_ref_str += f"Param {_+1}: [-30, 30]\n"
    md_ref_str += "```"
    reference_div = dbc.Card([dbc.CardHeader("Reference Range"),
                              dbc.CardBody([
                                  dcc.Markdown(md_ref_str)
                              ]),
                              ], 
                              className='mt-3',
                              style={'height': '400px'}
                              )

    # Latent Space
    latent_div = dbc.Card([dbc.CardHeader("Parameters"),
                           html.Span(id=prefix + "hidden-latent-space", children=init_hidden_space,
                                     className='d-none'),
                           dbc.CardBody([
                               html.Div(
                                   children=latent_space, id=prefix + 'output-latent')
                           ]),],
                           style={'height': '550px'})

    # Right Output Display area
    output_div = dbc.Card(
        [
            dbc.CardHeader("Output"),
            dbc.CardBody(
                [
                    dcc.Graph(
                        # style={ 'width': 200},
                        id=prefix + 'output-content'
                    ),
                    # html.Div(children=[], id=prefix + 'output-content'),
                ],
                className="d-flex justify-content-center", #d-flex justify-content-center
                style={'height': '400px'}
            ),
        ]

    )

    # '''Image Display'''
    # Click to sample input image

    @app.callback(
        [
            Output(component_id=prefix + 'input-content-id',
                   component_property='children'),
            Output(component_id=prefix + 'input-content-id-hidden',
                   component_property='children'),
            Output(prefix + 'input-content', 'figure'),
            Output(component_id=prefix + "hidden-latent-space",
                   component_property='children')
        ],
        [Input(component_id=prefix + 'sample-input', component_property='n_clicks')])
    def sample_input(n_clicks):
        # Randomly sample a data from the dataset
        input_id = random.randint(0, len(dataset_unscaled) - 1)
        img = dataset_unscaled[input_id]
        print(f'Airfoil: No. {input_id}')

        color = np.array(['Upper Edge', 'Lower Edge'], dtype=object)
        color = np.append(np.repeat(color, len(img)//4), ['Lower Edge'])

        # Display the original airfoil
        # print(img.shape)
        img = _graph_resize(img)
        graph_img = px.line(x=img[:, 0],
                            y=img[:, 1],
                            # width=300,
                            color=color,
                            labels={'x': 'x', 'y': 'y', 'color': 'Edge'},
                            height=300,
                            template='plotly_white',
                            title='Original Airfoil')
        # Set the x and y axis to be the same scale
        graph_img.update_layout(
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
        )

        hx = data_pca[input_id]

        return "Input ID: " + str(input_id), input_id, graph_img, hx

    # Sliders
    #
    for slider_id in range(latent_size):
        @app.callback(
            Output(component_id=prefix + 'latent-slider-' +
                   str(slider_id), component_property="value"),
            [Input(component_id=prefix + "hidden-latent-space",
                   component_property='children')],
            [State(component_id=prefix + 'latent-slider-' + str(slider_id), component_property="id")])
        def set_latent_slider(latent_space, slider_id):
            # print(f"Latent: {latent_space}")
            # print(f"Latent: {type(latent_space)}")
            # print(f"Slider ID: {slider_id}")
            # print(f"Slider ID: {type(slider_id)}")
            slider_id = int(slider_id.split("-")[-1])
            return float(latent_space[slider_id])

    # change output based on the latent space
    @app.callback(
        Output(component_id=prefix + 'output-content',
               component_property='figure'),
        [Input(component_id=prefix + 'latent-slider-' + str(slider_id), component_property='value')
         for slider_id in range(latent_size)],
        State(component_id=prefix + 'input-content-id-hidden',
              component_property='children')
    )
    def predicted_output(*params):
        # x value for the vertical line
        x_vline = 0.3
        # Unpack the parameters
        latent_space = np.array(params[:-1])
        input_id = params[-1]
        # Original
        original_img = _graph_resize(dataset_unscaled[input_id]) 
        # Altered Result
        print("Latent Space: ", [f"{i:.2f}" for i in latent_space])
        result_data = np.dot(latent_space, model.components_) + model.mean_
        result_data = result_data.reshape(1, -1)
        result_data = pre_process.inverse_transform(result_data)
        result_data = result_data.reshape(-1)
        result_img = _graph_resize(result_data)

        # Only show the part of the original where x > x_vline
        original_img = original_img[original_img[:, 0] > x_vline]
        # Only show the part of the result where x < x_vline
        result_img = result_img[result_img[:, 0] < x_vline]

        # reverse the first half in original_img
        original_img[:len(original_img)//2] = original_img[:len(original_img)//2][::-1]
        # reverse the second half in original_img
        original_img[len(original_img)//2:] = original_img[len(original_img)//2:][::-1]
        
        #* Adjust the result image so the two edges are aligned
        # Find the index of the point with the smallest x value in the first half of the original image
        original_min_x_index = np.argmin(original_img[:len(original_img)//2, 0])
        # Find the index of the point with the smallest x value in the second half of the original image
        original_max_x_index = np.argmin(original_img[len(original_img)//2:, 0]) + len(original_img)//2
        # Find the difference of y values between the two points
        y_diff_original = original_img[original_min_x_index, 1] - original_img[original_max_x_index, 1]
        #? print(f'P1{original_img[original_min_x_index]}')
        #? print(f'P2{original_img[original_max_x_index]}')
        #? print(y_diff_original)
        # Find the index of the point with the largest x value in first half of the result image
        result_min_x_index = np.argmax(result_img[:len(result_img)//2, 0])
        # Find the index of the point with the largest x value in second half of the result image
        result_max_x_index = np.argmax(result_img[len(result_img)//2:, 0]) + len(result_img)//2
        # Find the difference of y values between the two points
        y_diff_result = result_img[result_min_x_index, 1] - result_img[result_max_x_index, 1]
        #? print(f'P1{result_img[result_min_x_index]}')
        #? print(f'P2{result_img[result_max_x_index]}')
        #? print(y_diff_result)
        # Calculate Scale Ratio
        scale_ratio = y_diff_original / abs(y_diff_result)
        #? print(scale_ratio)

        # scale the result image
        result_img[:, 1] *= scale_ratio
        
        if result_img[result_min_x_index, 1] < result_img[result_max_x_index,1]:
            y_diff = original_img[original_max_x_index, 1] - result_img[result_min_x_index, 1]
            result_img[:, 1] += y_diff
        else:
            y_diff = original_img[original_min_x_index, 1] - result_img[result_min_x_index, 1]
            result_img[:, 1] += y_diff
        
        # create the color array for original_img, The first half is Upper Edge, the second half is Lower Edge
        color = np.array(['Upper Edge', 'Lower Edge'], dtype=object)
        color = np.repeat(color, len(original_img)//2)
        # Show Original Right part
        graph_img = px.line(x=original_img[:, 0],
                            y=original_img[:, 1],
                            color=color,
                            labels={'x': 'x', 'y': 'y', 'color': 'Edge'},
                            template='plotly_white',
                            title='Result')

        # Show Result Left part: Upper Edge
        graph_img.add_scatter( x=result_img[:len(result_img)//2, 0],
                                y=result_img[:len(result_img)//2, 1],
                                mode='lines',
                                name='Altered Upper',
                                )
        # set upper edge color to same as the original
        graph_img.update_traces(line_color='#636efa', selector=dict(name='Altered Upper'))
        
        # Show Result Left part: Lower Edge, add labels
        graph_img.add_scatter( x=result_img[len(result_img)//2:, 0],
                                y=result_img[len(result_img)//2:, 1],
                                mode='lines',
                                name='Altered Lower',
                                )
        # set lower edge color to same as the original
        graph_img.update_traces(line_color='#ef553b', selector=dict(name='Altered Lower'))
        
        # Set the x and y axis to be the same scale
        graph_img.update_layout(
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
        )
        # Set the range of y
        graph_img.update_yaxes(range=[-0.2, 0.2])
        # add a vertical indicator line, with text
        graph_img.add_vline(x=x_vline-0.006, line_width=2, line_dash="dash", line_color="green")
        # graph_img.add_annotation(x=x_vline+0.05, y=0.2, text=f"{x_vline}", showarrow=False, font=dict(color="green"))
        
        return graph_img

    def predicted_output2(*params):
        ## 先做编辑物理参数，直接由这个物理参数出发。


        

        output = model(input)
        return output
      

    '''Refresh the model when selected model changes'''

    pca_div = dbc.Card(
        [dbc.CardHeader(header),
         dbc.CardBody(dbc.Row([
            dbc.Col(latent_div, width=4),
            dbc.Col(input_div, width=8),
            dbc.Col(reference_div, width=4),
            dbc.Col(output_div, width=8), 
         ], className=''))],
        className="mt-4 mb-4 border-secondary autoencoder-box"
    )
    return pca_div
