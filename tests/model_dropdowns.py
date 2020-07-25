import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


app = dash.Dash()


def app_layout():
    ret = html.Div(
        id='div_models',
        children=[
            html.Div(
                id='div_current_model'
            ),
            dcc.Dropdown(
                id='dropdown_task_types',
                placeholder="Select type of task",
                options=[
                    {'label': 'Medical Image Semantic Segmentation',
                        'value': 'med_sem_seg'},
                    {'label': 'Natural Image Semantic Segmentation',
                        'value': 'nat_sem_seg'},
                    {'label': 'Medical Image Instance Segmentation',
                        'value': 'med_ins_seg', 'disabled': True},
                    {'label': 'Natural Image Instance Segmentation',
                        'value': 'nat_ins_seg', 'disabled': True},
                    {'label': 'Medical Image Object Detection',
                        'value': 'med_obj_det', 'disabled': True},
                    {'label': 'Natural Image Object Detection',
                        'value': 'nat_obj_det', 'disabled': True},
                ],
                multi=False
            ),

            dcc.Dropdown(
                id='dropdown_model_types',
                placeholder="Select type of model",
                options=[
                    {'label': '2D Models',
                        'value': '2d_models'},
                    {'label': '3D Models',
                        'value': '3d_models'},
                    {'label': 'Interactive Models',
                        'value': 'interactive_models'}
                ]
            ),
            dcc.Dropdown(
                id='dropdown_model_names',
                placeholder="Select name of model",
            ),

            html.Div(id='current_file')
        ]
    )
    return ret


app.layout = app_layout()


# callback
@app.callback(
    [
        Output('dropdown_model_names', 'options'),
        Output('current_file', 'children')
    ],
    
    [
        Input('dropdown_task_types', 'value'),
        Input('dropdown_model_types', 'value')
    ],
    [
        # State('dropdown_task_types', 'value'),
        # State('dropdown_model_types', 'value')
    ]
)
def select_task_and_model(task_type, model_type):
    options = []
    print(task_type, model_type)
    if task_type=='nat_sem_seg':
        if model_type=='2d_models':
            options = [
                {'label': 'fcn_resnet101', 'value': 'fcn_resnet101'},
                {'label': 'deeplabv3_resnet101', 'value': 'deeplabv3_resnet101'}
            ]        
    return options, ''


if __name__ == '__main__':
    app.run_server(debug=True, port=9878)
