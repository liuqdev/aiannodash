import json
import os
import time
import uuid
from copy import deepcopy
import csv
import sys
import pathlib
import base64
import glob

import boto3
import dash
from flask import Flask, send_from_directory
import dash_core_components as dcc
import dash_html_components as html
import requests
from dash.dependencies import Input, Output, State
from flask_caching import Cache

import dash_reusable_components as drc
import utils

#################################################################################
# Application Configuration
#################################################################################
DEBUG = True
LOCAL = False
APP_PATH = str(pathlib.Path(__file__).parent.resolve())

# Default app and server
# app = dash.Dash(__name__)
# server = app.server

# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server)

@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)



# https://docs.faculty.ai/user-guide/apps/examples/dash_file_upload_download.html
UPLOAD_DIRECTORY = "./app_uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
#################################################################################

def get_dropdown_datasets_list():
    ret_dropdown_datasets_list = []
    datasets_list = uploaded_dirs(UPLOAD_DIRECTORY, "/*")
    print(datasets_list)
    for dataset_fn in datasets_list:
        dataset_basename = os.path.basename(dataset_fn)
    ret_dropdown_datasets_list.append({'label': dataset_basename, 'value': dataset_basename})
    return ret_dropdown_datasets_list


def get_app_layout(session_id):
    app_layout = html.Div(
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
                                id="logo",
                                src=app.get_asset_url("dash-logo-new.png")),
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
                    dcc.Tabs([
                        # Open/Upload file(s)
                        # 打开或者上传数据
                        dcc.Tab(
                            label="Files",
                            children=[
                                drc.Card([
                                    # 选择上传的图像文件格式
                                    drc.NamedInlineRadioItems(
                                        name="Image Display Format",
                                        short="encoding-format",
                                        options=[
                                            {
                                                "label": "DICOM(.dcm)",
                                                "value": "dicom"
                                            },
                                            {
                                                "label": "NIFTI(.nii)",
                                                "value": "nifti"
                                            },
                                            {
                                                "label": "NIFTI(nii.gz)",
                                                "value": "niftigz"
                                            },
                                            {
                                                "label": "PNG(.png)",
                                                "value": "png"
                                            },
                                            {
                                                "label": "JPG(.jpg)",
                                                "value": "jpg"
                                            },
                                            {
                                                "label": "JPEG(.jpeg)",
                                                "value": "jpeg"
                                            },
                                            {
                                                "label": "Image file(s)",
                                                "value": "image"
                                            },
                                        ],
                                        val="png",
                                    ),
                                    # 文件上传
                                    dcc.Upload(
                                        id="upload-data",
                                        children=[
                                            "Drag and Drop or ",
                                            html.
                                            A(children="Select an Image/Images"
                                              ),
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
                                        #accept="image/png, image/jpeg, image/jpg",
                                        # accept='.dcm, .nii',
                                        accept="image/*",
                                        multiple=
                                        True,  # support multiple files uploading
                                    ),
                                    # 显示上传的文件列表
                                    html.Ul(id="ul_file_list", 
                                        style={
                                            "height":"100px",
                                            "overflow":"hidden",
                                            "overflow-y":"scroll"
                                        }
                                    ),

                                    # 输入数据集的名称
                                    # save input image(s) as an dataset
                                    dcc.Input(
                                        id='input_dataset_name',
                                        type='text',
                                        placeholder='Input dataset name'
                                    ),
                                    # 确认
                                    html.Button(id='btn_upload', n_clicks=0, children='Upload'),

                                    # drc.NamedInlineRadioItems(
                                    #     name="Selection Mode",
                                    #     short="selection-mode",
                                    #     options=[
                                    #         {"label": " Rectangular", "value": "select"},
                                    #         # {"label": " Lasso", "value": "lasso"},
                                    #     ],
                                    #     val="select",
                                    # ),
                                ]),
                                drc.Card([
                                    html.Div(
                                    id="file_menu",
                                    children=[
                                        
                                        html.Div(
                                            id='div_current_dataset'
                                        ),

                                        # 数据集名称
                                        # Dataset slection
                                        dcc.Dropdown(
                                            id='dropdown_datasets_list',
                                            
                                            style={
                                                # 'width': '50%',
                                                # 'display': 'inline-block'
                                            },
                                            placeholder="Select a dataset",
                                        ),

                                        html.Div(
                                            id='div_current_file'
                                        ),
                                        # 选择文件
                                        # file selection
                                        dcc.Dropdown(
                                            id='dropdown_files_list',
                                            # options=[{
                                            #     'label': '',
                                            #     'value': ''
                                            # }],
                                            style={
                                                # 'width': '50%',
                                                # 'display': 'inline-block'
                                            },
                                            placeholder="Select a file",
                                        )
                                    ]),
                                ]),

                                # dcc.Graph(
                                #     id="graph-histogram-colors",
                                #     figure={
                                #         "layout": {
                                #             "paper_bgcolor": "#272a31",
                                #             "plot_bgcolor": "#272a31",
                                #         }
                                #     },
                                #     config={"displayModeBar": False},
                                # ),

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
                            ]),
                        dcc.Tab(
                            label="Tools",
                            children=[
                                drc.Card([
                                    drc.CustomDropdown(
                                        id="dropdown-filters",
                                        options=[
                                            {
                                                "label": "Blur",
                                                "value": "blur"
                                            },
                                            {
                                                "label": "Contour",
                                                "value": "contour"
                                            },
                                            {
                                                "label": "Detail",
                                                "value": "detail"
                                            },
                                            {
                                                "label": "Enhance Edge",
                                                "value": "edge_enhance"
                                            },
                                            {
                                                "label": "Enhance Edge (More)",
                                                "value": "edge_enhance_more",
                                            },
                                            {
                                                "label": "Emboss",
                                                "value": "emboss"
                                            },
                                            {
                                                "label": "Find Edges",
                                                "value": "find_edges"
                                            },
                                            {
                                                "label": "Sharpen",
                                                "value": "sharpen"
                                            },
                                            {
                                                "label": "Smooth",
                                                "value": "smooth"
                                            },
                                            {
                                                "label": "Smooth (More)",
                                                "value": "smooth_more"
                                            },
                                        ],
                                        searchable=False,
                                        placeholder="Basic Filter...",
                                    ),
                                    drc.CustomDropdown(
                                        id="dropdown-enhance",
                                        options=[
                                            {
                                                "label": "Brightness",
                                                "value": "brightness"
                                            },
                                            {
                                                "label": "Color Balance",
                                                "value": "color"
                                            },
                                            {
                                                "label": "Contrast",
                                                "value": "contrast"
                                            },
                                            {
                                                "label": "Sharpness",
                                                "value": "sharpness"
                                            },
                                        ],
                                        searchable=False,
                                        placeholder="Enhance...",
                                    ),
                                    html.Div(
                                        id="div-enhancement-factor",
                                        children=[
                                            f"Enhancement Factor:",
                                            html.Div(children=dcc.Slider(
                                                id="slider-enhancement-factor",
                                                min=0,
                                                max=2,
                                                step=0.1,
                                                value=1,
                                                updatemode="drag",
                                            )),
                                        ],
                                    ),
                                    html.Div(
                                        id="button-group",
                                        children=[
                                            html.Button(
                                                "Run Operation",
                                                id="button-run-operation"),
                                            html.Button("Undo",
                                                        id="button-undo"),
                                        ],
                                    ),
                                ]),
                            ]),
                        dcc.Tab(label="Tab three", children=[]),
                    ]),
                ],
            ),
        ],
    )
    return app_layout


def serve_layout():
    # Generates a session ID
    session_id = str(uuid.uuid4())

    # Post the image to the right key, inside the bucket named after the
    # session ID
    # store_image_string(utils.IMAGE_STRING_PLACEHOLDER, session_id)

    # App Layout
    app_layout = get_app_layout(session_id)
    return app_layout


app.layout = serve_layout


# @app.callback(
#     Output('dropdown_files_list', 'options'),
#     [Input('upload-data', 'filename'),
#      Input('upload-data', 'contents')])
# def update_output(uploaded_filenames, uploaded_file_contents):
#     files = []
#     if uploaded_filenames is not None and uploaded_file_contents is not None:
#         for name, data in zip(uploaded_filenames, uploaded_file_contents):
#             print(name)
#             files.append(name)
#     # files = uploaded_files()
#     # files = uploaded_files()
#     print("--------", len(files))
#     if len(files) == 0:
#         return [{'label': '', 'value': ''}]
#     else:
#         return [{"label": filename, "value": filename} for filename in files]


# 选择上传的数据类型, 确保每次上传的数据集(作为模型的输入)是类型一致的
@app.callback(Output('upload-data', 'accept'),
              [Input('radio-encoding-format', 'value')])
def select_upload_format(item):
    accept = 'image/*'
    if item == 'image':
        accept = "image/*"
    elif item == 'dicom':
        accept = ".dcm"
    elif item == 'nifti':
        accept = ".nii"
    elif item == 'niftigz':
        accept = '.nii.gz'
    elif item == 'png':
        accept = '.png'
    elif item == 'jpg':
        accept = '.jpg'
    elif item == 'jpeg':
        accept = '.jpeg'
    return accept


# 上传文件作为一个数据集
@app.callback(
    [
        Output('ul_file_list', 'children'),
        Output('dropdown_datasets_list', 'options'),
        # Output('dropdown_datasets_list', 'value'),
        # Output('dropdown_files_list', 'options')
    ],
    [
        Input('btn_upload', 'n_clicks'),  # 确认数据集名称
        Input('upload-data', 'filename'),
        Input('upload-data', 'contents'),
    ],
    [
        # State('upload-data', 'filename'),
        # State('upload-data', 'contents'),
        State('input_dataset_name', 'value')  # 输入数据集名称
    ]
)
def upload_dataset(n_clicks, filenames, contents, dataset_name):
    print('dataset name', dataset_name)
    if filenames is not None and contents is not None:
        for fn in filenames:
            print(f'filename: {fn}')

        ret_ul_file_list = [html.Li(filename) for filename in filenames]
    else:
        ret_ul_file_list = [html.Li("No files yet!")]
    
    ret_dropdown_datasets_list = []
    datasets_list = uploaded_dirs(UPLOAD_DIRECTORY, "/*")
    for dataset_fn in datasets_list:
        dataset_basename = os.path.basename(dataset_fn)
        ret_dropdown_datasets_list.append({'label': dataset_basename, 'value': dataset_basename})

    if n_clicks>0:
        print("uploading ...", filenames)
        print("n_clicks", n_clicks)
        # 将上传的文件保存到本地以dataset_name命名的路径下
        dataset_dir = os.path.join(UPLOAD_DIRECTORY, dataset_name)
        mkdir(dataset_dir)
        print('save image(s) to {}'.format(dataset_dir))
        for fn, dat in zip(filenames, contents):
            save_file(os.path.join(dataset_dir, fn), dat)       

        ret_dropdown_datasets_list.append({'label': dataset_name, 'value': dataset_name})
        # ret_dropdown_files_list = [{"label": filename, "value": filename} for filename in filenames]
        print('dataset_name', dataset_name)
        return ret_ul_file_list , ret_dropdown_datasets_list #, dataset_name
    else:
        return ret_ul_file_list , ret_dropdown_datasets_list #, []


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
            

# 原样保存文件
def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(name, "wb") as fp:
        fp.write(base64.decodebytes(data))

# 显示已经上传的数据集的名称
def uploaded_dirs(path, pattern="/*"):
    dirs_fns = []
    for fn in glob.glob(path+pattern):
        if os.path.isdir(fn):
            dirs_fns.append(fn)
    return dirs_fns

# 显示已经上传的文件夹中的文件
def uploaded_files(path, pattern="/*"):
    """List the files in the upload directory."""
    files_fns = []
    for filename in glob.glob(path+pattern):
        if os.path.isfile(filename):
            files_fns.append(filename)
    return files_fns

# 创建下载链接
def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    print("location: ", location)
    return html.A(filename, href=location)


# # 根据数据集名称, 更新图像数据列表和下拉列表
# @app.callback(
#     [
#         Output('dropdown_files_list', 'options'),
#         Output('ul_file_list', 'children')
#     ],
#     [
#         Input('dropdown_datasets_list', 'value')
#     ]
# )
# def select_dataset(dataset_name):
#     if dataset_name is not None:
#         print("----------------", dataset_name)
#         ret_ul_file_list = []
#         ret_dropdown_files_list = []

#         files_fns = uploaded_files(os.path.join(UPLOAD_DIRECTORY, dataset_name))
        
#         for filename in files_fns:
#             base_name = os.path.basename(filename)

#             ret_dropdown_files_list.append({"label": base_name, "value": base_name})
#             ret_ul_file_list.append(html.Li(base_name))
            
#         return ret_dropdown_files_list, ret_ul_file_list
#     else:
#         return [], []

@app.callback(
    [
        Output('div_current_dataset', 'children'),
        Output('dropdown_files_list', 'options'),
        Output('div_current_file', 'children'),
    ],
    [
        Input('dropdown_datasets_list', 'value'),
        Input('dropdown_files_list', 'value')
    ],
    # [
    #     State('dropdown_files_list', 'value')
    # ]
)
def display_current_dataset(dataset_name, file_name):
    if dataset_name is not None:
        ret_dropdown_files_list = []
        files_fns = uploaded_files(os.path.join(UPLOAD_DIRECTORY, dataset_name))
        for filename in files_fns:
            base_name = os.path.basename(filename)
            ret_dropdown_files_list.append({"label": base_name, "value": base_name})
            
        # return ret_dropdown_files_list, ret_ul_file_list
        return 'current dataset name: {}'.format(dataset_name), ret_dropdown_files_list, 'current file name: {}'.format(file_name)
    else:
        return 'current dataset name: {}'.format(dataset_name), [], 'current file name: {}'.format(file_name)




# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=9997)
