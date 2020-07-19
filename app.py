import json
import os
import time
import uuid
from copy import deepcopy
import csv
import sys
import pathlib

import boto3
import dash
import dash_core_components as dcc
import dash_html_components as html
import requests
from dash.dependencies import Input, Output, State
from flask_caching import Cache

import dash_reusable_components as drc
import utils

DEBUG = True
LOCAL = False
APP_PATH = str(pathlib.Path(__file__).parent.resolve())

app = dash.Dash(__name__)
server = app.server

def serve_layout():
    # Generates a session ID
    session_id = str(uuid.uuid4())

    # Post the image to the right key, inside the bucket named after the
    # session ID
    # store_image_string(utils.IMAGE_STRING_PLACEHOLDER, session_id)

    # App Layout
    return html.Div(
        id="root",
        children=[

            # Session ID
            html.Div(session_id, id="session-id"),

            # Main body
            html.Div(
                id="app-container",
                children=[
                    # Banner display
                    html.Div(
                        id="banner",
                        children=[
                            html.Img(
                                id="logo", src=app.get_asset_url("dash-logo-new.png")
                            ),
                            html.H2("AI Annotation App", id="title"),
                        ],
                    ),
                    # Graph
                    html.Div(
                        id="image",
                        children=[
                            # The Interactive Image Div contains the dcc Graph
                            # showing the image, as well as the hidden div storing
                            # the true image
                            html.Div(
                                id="div-interactive-image",
                                children=[
                                    # utils.GRAPH_PLACEHOLDER,
                                    # html.Div(
                                    #     id="div-storage",
                                    #     children=utils.STORAGE_PLACEHOLDER,
                                    # ),

                                    # file list dropdown
                                    html.Div(
                                        id="file-list",
                                        children=[
                                            # "Slect a file: ",
                                            dcc.Dropdown(
                                                id='filelist_dropdown',
                                                options=[
                                                    {'label':'', 'value': ''}
                                                ],
                                                placeholder="Select a file",
                                                style={'width': '50%', 'display': 'inline-block'}
                                            )
                                        ]
                                    ),
                                    # graph area
                                    # utils.GRAPH_PLACEHOLDER,
                                ],
                            )
                        ],
                    ),
                ],
            ),


            # Sidebar
            html.Div(
                id="sidebar",
                children=[
                    dcc.Tabs(
                        [
                            dcc.Tab(
                                label="Files",
                                children=[
                                    drc.Card(
                                        [
                                            dcc.Upload(
                                                id="upload-data",
                                                children=[
                                                    "Drag and Drop or ",
                                                    html.A(children="Select an Image/Images"),
                                                ],
                                                # No CSS alternative here
                                                style={
                                                    "color": "darkgray",
                                                    "width": "100%",
                                                    "height": "50px",
                                                    "lineHeight": "50px",
                                                    "borderWidth": "1px",
                                                    "borderStyle": "dashed",
                                                    "borderRadius": "5px",
                                                    "borderColor": "darkgray",
                                                    "textAlign": "center",
                                                    "padding": "2rem 0",
                                                    "margin-bottom": "2rem",
                                                },
                                                # accept="image/*",
                                                multiple=True,  # support multiple files uploading
                                            ),
                                            # drc.NamedInlineRadioItems(
                                            #     name="Selection Mode",
                                            #     short="selection-mode",
                                            #     options=[
                                            #         {"label": " Rectangular", "value": "select"},
                                            #         # {"label": " Lasso", "value": "lasso"},
                                            #     ],
                                            #     val="select",
                                            # ),
                                            drc.NamedInlineRadioItems(
                                                name="Image Display Format",
                                                short="encoding-format",
                                                options=[
                                                    {"label": " PNG", "value": "png"},
                                                    {"label": " JPEG", "value": "jpeg"},
                                                    {"label": " DICOM", "value": "dicom"},
                                                    {"label": " NIFTI", "value": "nifti"},
                                                ],
                                                val="png",
                                            ),
                                        ]
                                    ),
                                    dcc.Graph(
                                        id="graph-histogram-colors",
                                        figure={
                                            "layout": {
                                                "paper_bgcolor": "#272a31",
                                                "plot_bgcolor": "#272a31",
                                            }
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                    # dcc.Graph(
                                    #     id="graph-histogram-colors2",
                                    #     figure={
                                    #         "layout": {
                                    #             "paper_bgcolor": "#272a31",
                                    #             "plot_bgcolor": "#272a31",
                                    #         }
                                    #     },
                                    #     config={"displayModeBar": False},
                                    # ),
                                ]
                            ),

                            dcc.Tab(
                                label="Tab two",
                                children=[
                                    drc.Card(
                                        [
                                            drc.CustomDropdown(
                                                id="dropdown-filters",
                                                options=[
                                                    {"label": "Blur", "value": "blur"},
                                                    {"label": "Contour", "value": "contour"},
                                                    {"label": "Detail", "value": "detail"},
                                                    {"label": "Enhance Edge", "value": "edge_enhance"},
                                                    {
                                                        "label": "Enhance Edge (More)",
                                                        "value": "edge_enhance_more",
                                                    },
                                                    {"label": "Emboss", "value": "emboss"},
                                                    {"label": "Find Edges", "value": "find_edges"},
                                                    {"label": "Sharpen", "value": "sharpen"},
                                                    {"label": "Smooth", "value": "smooth"},
                                                    {"label": "Smooth (More)", "value": "smooth_more"},
                                                ],
                                                searchable=False,
                                                placeholder="Basic Filter...",
                                            ),
                                            drc.CustomDropdown(
                                                id="dropdown-enhance",
                                                options=[
                                                    {"label": "Brightness", "value": "brightness"},
                                                    {"label": "Color Balance", "value": "color"},
                                                    {"label": "Contrast", "value": "contrast"},
                                                    {"label": "Sharpness", "value": "sharpness"},
                                                ],
                                                searchable=False,
                                                placeholder="Enhance...",
                                            ),
                                            html.Div(
                                                id="div-enhancement-factor",
                                                children=[
                                                    f"Enhancement Factor:",
                                                    html.Div(
                                                        children=dcc.Slider(
                                                            id="slider-enhancement-factor",
                                                            min=0,
                                                            max=2,
                                                            step=0.1,
                                                            value=1,
                                                            updatemode="drag",
                                                        )
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="button-group",
                                                children=[
                                                    html.Button(
                                                        "Run Operation", id="button-run-operation"
                                                    ),
                                                    html.Button("Undo", id="button-undo"),
                                                ],
                                            ),
                                        ]
                                    ),
                                ]
                            ),

                            dcc.Tab(
                                label="Tab three",
                                children=[

                                ]
                            ),
                        ]
                    ),
                ],
            ),
        ],
    )


app.layout = serve_layout

@app.callback(
    Output('filelist_dropdown', 'options'),    
    [Input('upload-data', 'filename'),
    Input('upload-data', 'contents')]
)
def update_output(uploaded_filenames, uploaded_file_contents):
    files = []
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            print(name)
            files.append(name)
    # files = uploaded_files()
    # files = uploaded_files()
    print("--------", len(files))
    if len(files) == 0:
        return [{'label': '', 'value': ''}]
    else:
        return [{"label": filename, "value": filename} for filename in files]


@app.callback(
    Output('', ''),
    [Input('', '')]
)
def show_a_2D_image(filename):
    pass

# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=9999)

