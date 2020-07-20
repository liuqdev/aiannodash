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
import requests

import numpy as np
import pydicom
import imageio
import cv2
import nibabel as nib
import boto3
import dash
from flask import Flask, send_from_directory
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
from flask_caching import Cache

import dash_reusable_components as drc
import utils
from utils import numpy_to_b64

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

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll',
        "height":"100px",
    },
    'Ul':{
        "height":"100px",
        "overflow":"hidden",
        "overflow-y":"scroll"
    }
}

# https://docs.faculty.ai/user-guide/apps/examples/dash_file_upload_download.html
UPLOAD_DIRECTORY = "./app_uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
#################################################################################


##############################################################################################
# Helper functions
##############################################################################################
def add_img_to_figure(image_code, height=600, width=1600, scale_factor=1):
    # Create figure
    fig = go.Figure()

    # Constants
    img_width = width
    img_height = height
    scale_factor = scale_factor

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=True,
        #range=[0, img_height * scale_factor],
        range=[img_height * scale_factor, 0],  # 调整坐标范围用
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
        x=0,
        sizex=img_width * scale_factor,
        #y=img_height * scale_factor,
        y=0,
        sizey=img_height * scale_factor,
        xref="x",
        yref="y",
        opacity=1.0,
        layer="below",
        sizing="stretch",
        source=image_code
        )
    )

    # Configure other layout
    fig.update_layout(
        # clickmode='event+select',
        # width=img_width * scale_factor,
        # height=img_height * scale_factor,
        # width=600,
        height=700,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#272a31",
        plot_bgcolor="#272a31",
        # autosize=True
    )
    return fig

def get_dropdown_datasets_list():
    ret_dropdown_datasets_list = []
    datasets_list = uploaded_dirs(UPLOAD_DIRECTORY, "/*")
    print(datasets_list)
    for dataset_fn in datasets_list:
        dataset_basename = os.path.basename(dataset_fn)
    ret_dropdown_datasets_list.append({'label': dataset_basename, 'value': dataset_basename})
    return ret_dropdown_datasets_list

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


def read_file(file_fn, file_format):
    """
    Support format: jpg/jpeg, png, dcm, nii, nii.gz, npy
    """
    import numpy as np
    d = None  # data return
    try:
        print(file_fn, "-----------------", file_format)
        if file_format.lower() in {"png", "jpeg", "jpg", "bmp"}:
            d = np.array(imageio.imread(file_fn))
            print("=================", type(d))
        elif file_format.lower() in {"dcm"}:
            # import pydicom
            ds = pydicom.read_file(file_fn)
            # img = ds.pixel_array
            d = ds
        elif file_format.lower() in {"nii", 'nii.gz'}:  # 3D/4D files
            import nibabel as nib  
            d = nib.load(file_fn)
        elif file_format.lower() in {'npy'}:
            
            d = np.load(file_fn)
        else:
            # TODO: 使用medoy支持更多类型的医学数据
            print("Unsupport image format")
        return d
    except Exception as e:
        print("File Reading error !", e)

# 根据文件名来读取文件
def get_image_data(file_fn, file_format, enc_format='png'):
    
    data = {
        'fullname': file_fn,
        'format': file_format
    }

    img_decoded = None
    
    # 对于不同类型的数据分别处理
    d = read_file(file_fn, file_format)
    # eg. xxx.png, shape(h, w)
    if isinstance(d, np.ndarray):
        # 灰度图
        # gray image
        if d.ndim==2:
            img = d
            h, w = d.shape[0], d.shape[1]
            if_scalar = False if img.dtype==np.uint8 else True
            img_encoded = numpy_to_b64(d, enc_format=enc_format, scalar=if_scalar)
            img_decoded = 'data:image/{};base64,{}'.format(enc_format, img_encoded)
            data['ndim'] = 2
            data['width'] = w
            data['height'] = h
            data['encoded_b64'] = img_encoded
            data['decoded_b64'] = img_decoded
        elif d.ndim==3:
            # 可能是彩图等多通道图, 也可能是3D图
            if d.shape[2] in {1, 3, 4}:
                img = d.squeeze()
                h, w = d.shape[0], d.shape[1]
                if_scalar = False if img.dtype==np.uint8 else True
                img_encoded = numpy_to_b64(d, enc_format=enc_format, scalar=if_scalar)
                img_decoded = 'data:image/{};base64,{}'.format(enc_format, img_encoded)
                data['ndim'] = 3
                data['width'] = w
                data['height'] = h
                data['encoded_b64'] = img_encoded
                data['decoded_b64'] = img_decoded
                print("--------------------3")
            else:  # 3D图
                pass 
            pass
    elif isinstance(d, str):  # 3D or 4D图
        pass

    return data
#########################################################################################

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
                                id="div-interactive_image",
                                children=[
                                    # utils.GRAPH_PLACEHOLDER,

                                    # main graph
                                    # 主要的显示区
                                    dcc.Graph(
                                        id="interactive_image",
                                        
                                    ),

html.Div(
    id='div_slicer'
),

                                    # dcc.Slider(
                                    #     id="slicer_images",
                                    #         min=0,
                                    #         max=100,
                                    #         value=65,
                                    #         marks={
                                    #             0: {'label': '0 °C', 'style': {'color': '#77b0b1'}},
                                    #             26: {'label': '26 °C'},
                                    #             37: {'label': '37 °C'},
                                    #             100: {'label': '100 °C', 'style': {'color': '#f50'}}
                                    #         }
                                    # ),


                                    html.Img(
                                        id="img_show",
                                    ),

                                    html.Div(
                                        id="div-storage",
                                        children=utils.STORAGE_PLACEHOLDER,
                                    ),
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
                                        style=styles['Ul']
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
                                    # drc.NamedInlineRadioItems(
                                    #     name="Selection Mode",
                                    #     short="selection-mode",
                                    #     options=[
                                    #         {"label": " Rectangular", "value": "select"},
                                    #         {"label": " Lasso", "value": "lasso"},
                                    #     ],
                                    #     val="select",
                                    # ),
                                    # 选择的数据
                                    html.Pre(id='selected-data', style=styles['pre']),
                                    # 选择的区域
                                    html.Pre(id='relayout-data', style=styles['pre']),
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
    print('func: upload_dataset', dataset_name)
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

# 选择具体数据集的图像文件
@app.callback(
    [
        Output('div_current_dataset', 'children'),
        Output('dropdown_files_list', 'options'),
        Output('div_current_file', 'children'),        
        Output('img_show', 'src'),
        Output('interactive_image', 'figure'),
        Output('div_slicer', 'children')
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
    ret = [
        'current dataset name: {}'.format(dataset_name),
        [],
        'current file name: {}'.format(file_name),
        '',
        utils.FIGURE_PLACEHOLDER,  # 默认显示
        dcc.Slider(disabled=True)
    ]

    print("func display_current_dataset")
    print(dataset_name, file_name)
    if dataset_name is not None:
        ret_dropdown_files_list = []
        files_fns = uploaded_files(os.path.join(UPLOAD_DIRECTORY, dataset_name))
        for filename in files_fns:
            base_name = os.path.basename(filename)
            ret_dropdown_files_list.append({"label": base_name, "value": base_name})
        
        if file_name is not None:
            # 读取当前文件, 将其输出到interactive_image上
            current_file = os.path.join(UPLOAD_DIRECTORY, dataset_name, file_name)
            file_format = '.'.join(file_name.split('.')[1:])
            print(current_file, file_format)

            img_data = get_image_data(file_fn=current_file, file_format=file_format, enc_format='png')
            print(img_data.keys())
            
            if 'decoded_b64' in img_data.keys():
                ret[3] = img_data['decoded_b64']
                fig = add_img_to_figure(img_data['decoded_b64'], height=img_data['height'], width=img_data['width'])
                ret[4] = fig

                idx = files_fns.index(current_file)
                
                num_imgs = len(files_fns)
                step  = 1 if num_imgs<101 else num_imgs//10
                print("current file:{} idx: {}/{}".format(current_file, idx, num_imgs))
                slider_images = dcc.Slider(
                    id="slicer_images",
                        min=1,
                        max=num_imgs,
                        value=idx+1,
                        step=step,
                        marks={str(i+1): str(i+1) for i in range(0, num_imgs, 1)}
                )
                print({str(i+1): str(i+1) for i in range(0, num_imgs, 1)})
                ret[5] = slider_images


        # return ret_dropdown_files_list, ret_ul_file_list
        ret[1] = ret_dropdown_files_list
        
        return ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]  #, fig
    else:
        return ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]  #, go.Figure()

# @app.callback(
#     Output("interactive_image", "figure"),
#     [Input("radio-selection-mode", "value")],
#     [State("interactive_image", "figure")],
# )
# def update_selection_mode(selection_mode, figure):
#     if figure:
#         figure["layout"]["dragmode"] = selection_mode
#     return figure

@app.callback(
    Output('selected-data', 'children'),
    [Input('interactive_image', 'selectedData')])
def display_selected_data(selectedData):
    # print(type(selectedData))
    # print("display_selected_data")
    # print(selectedData)
    return json.dumps(selectedData, indent=2)


@app.callback(
    Output('relayout-data', 'children'),
    [Input('interactive_image', 'relayoutData')])
def display_relayout_data(relayoutData):
    # print("display_relayout_data")
    return json.dumps(relayoutData, indent=2)


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=9997)
