import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
import dash_table
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import dash_reusable_components as drc
from dash_reusable_components import numpy_to_b64

app = dash.Dash(__name__)

# 1 选择一个dcm结尾的文件
# 2 ~~提取其中的dataset和编号信息~~, 提取该文件的路径信息
# 读取该路径下的dicom文件(这里约定, 一个文件夹下, 最多只能
# 有一个case的dicom文件)
# 3 ~~遍历同级文件下和所选文件处于相同数据集和编号的文件~~
# 4 ~~合并为一个dcm进行处理~~

def get_ext(path):
    ext_name = ".".join(os.path.basename(path).split('.')[1:])
    return ext_name

# def read_dicom_series(series_folder):
#     print("Reading Dicom directory:", series_folder)
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(sys.argv[1])
#     reader.SetFileNames(dicom_names)

#     image = reader.Execute()

def get_metadata(ds):
    data_dict = {
        'name': [],
        'key': [],
        'VR': [],
        'VM': [],
        'value': []
    }
    for k, v in ds.items():
        data_dict['key'].append(str(k))
        x = pydicom.dataelem.DataElement_from_raw(v) if isinstance(v, pydicom.dataelem.RawDataElement) else v
        data_dict['name'].append(str(x.name))
        data_dict['value'].append(str(x.value))
        data_dict['VR'].append(str(x.VR))
        data_dict['VM'].append(str(x.VM))
        df = pd.DataFrame(data_dict)
        df = df[['key', 'name', 'VR', 'VM', 'value']]
    # df.head()
    return df

# https://dash.plotly.com/datatable
def get_app_layout():
    # df = pd.read_csv(r"D:\home\intern_project\AIAnnoLab\aiannodash\tests\solar.csv")
    df = pd.read_csv(r"D:\home\intern_project\AIAnnoLab\aiannodash\tests\dicom_sample_metadata.csv")
    print(df.head())
    
    # sample_path = r"D:\home\intern_project\AIAnnoLab\aiannodash\tests\dicom_series\i0000,0000b.dcm"
    # # sample_path = "dicom_series/i0000,0000b.dcm"  # 模拟下拉列表被选择
    # ext = get_ext(sample_path)  # 获取文件的扩展名, 这里支持多扩展名, 如nii.gz
    # dirname = os.path.dirname(sample_path)
    # if ext.lower()=='dcm':
    #     ds = pydicom.read_file(sample_path)
    #     df = get_metadata(ds)
    #     df = df.head(10)
    
    print(df)
    # print(type(df.to_dict('records')))
    data = df.head(10).to_dict('records')
    
    app_layout = html.Div(
        children=[
            dash_table.DataTable(
                    id='table_metadata',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=data,
                ),
            html.Button(    
                    "Diglog",
                    id='btn_dialog'
                ),

                html.Div(
                    id='dialog'
                )
            ]
        )
    return app_layout


# @app.callback(
#     Output('table_metadata', 'data'),
#     [
#         Input('btn_dialog', 'n_clicks')
#     ]
# )
# def popup_dialog(n_clicks):
#     if n_clicks:
#         sample_path = r"D:\home\intern_project\AIAnnoLab\aiannodash\tests\dicom_series\i0000,0000b.dcm"
#         # sample_path = "dicom_series/i0000,0000b.dcm"  # 模拟下拉列表被选择
#         ext = get_ext(sample_path)  # 获取文件的扩展名, 这里支持多扩展名, 如nii.gz
#         dirname = os.path.dirname(sample_path)
#         if ext.lower()=='dcm':
#             ds = pydicom.read_file(sample_path)
#             df = get_metadata(ds)
#             df = df.head()
#         print(df)
#         print(type(df.to_dict('records')))
#         return df.to_dict('records')
#     else:
#         df = pd.read_csv(r"D:\home\intern_project\AIAnnoLab\aiannodash\tests\solar.csv")
#         # df = pd.read_csv(r"D:\home\intern_project\AIAnnoLab\aiannodash\tests\dicom_sample_metadata.csv")
#         print(df.head())
#         print(type(df.to_dict('records')))
#         return df.to_dict('records')
        

if __name__ == '__main__':
    app.layout = get_app_layout()
    app.run_server(debug=True, port=9878)
