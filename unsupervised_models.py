from PIL import Image
import numpy as np
from skimage.morphology import erosion, dilation
from skimage import measure
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import numpy_to_b64

# reference: https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
def kmeans_seg(img, zone=None):
    img = np.array(img)
    img = Image.fromarray(img).convert('L')  # convert to gray image
    img = np.array(img)
    
    data = {}
    data['image_gray'] = img
    #Standardize the pixel values
    row_size= img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    data['image_normalized'] = img

    # Find the average pixel value near the lungs
    # to renormalize washed out images
    if zone==None:
        # 认为默认的中心
        middle = img[int(col_size/8):int(col_size/8*7), int(row_size/8):int(row_size/8*7)]
    else:
        assert isinstance(zone, tuple) and len(zone)==4
        lower, upper, left, right = zone
        middle = img[lower:upper+1, left:right+1]

    mean = np.mean(middle)  
    max_ = np.max(img)
    min_ = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max_]=mean
    img[img==min_]=mean

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0) 
    # data['image_threshold'] = thresh_img

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = erosion(thresh_img,np.ones([3,3]))
    dilated = dilation(eroded,np.ones([8, 8]))
    data['image_dilated'] = dilated
    labels = measure.label(dilated)  # Different labels are displayed in different colors
    data['image_labels'] = labels

    label_vals = np.unique(labels)
    data['label_values'] = label_vals

    # 区域增长算法
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0] < row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/10 and B[2]<col_size/10*9:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    for N in good_labels:
        mask = mask + np.where(labels==N, 1 ,0)
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    # mask = dilation(mask, np.ones([10,10])) # one last dilation

    data['mask'] = mask
    applied = mask*img
    data['mask_applied'] = applied
    
    # subplots_titles = ['Original', 'Threshold', 'After Erosion and Dilation', 'Color Labels', 'Final Mask', 'Apply Mask on Original']
    return data


def get_subplots_fig(rows, cols, imgs, types=None):
    # 需要保证输入的都是rgb, 0-255的numpy图像
    fig = make_subplots(
        rows, cols,
        horizontal_spacing=0,
        vertical_spacing=0,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for i in range(1, rows+1):
        for j in range(1, cols+1):
            idx = (i-1)*cols+j
            print(i, j, idx)
            im = imgs[idx-1]

            img_height, img_width = im.shape[0], im.shape[1]
            scale_factor = 1
            
            enc_format = 'png'
            b64 = numpy_to_b64(im, enc_format=enc_format, scalar=False)
            decoded = 'data:image/{};base64,{}'.format(enc_format, b64)
            
            fig.update_xaxes(
                visible=False,
                range=[0, img_width * scale_factor]
            )

            fig.update_yaxes(
                visible=False,
                #range=[0, img_height * scale_factor],
                range=[img_height * scale_factor, 0],  # 调整坐标范围用
                # the scaleanchor attribute ensures that the aspect ratio stays constant
                scaleanchor="x"
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, img_width * scale_factor],
                    y=[0, img_height * scale_factor],
                    mode="markers",
                    marker_opacity=0
                )
            )
            fig.add_layout_image(
                row=i,
                col=j,
                source=decoded,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=im.shape[1],
                sizey=im.shape[0],
                sizing="stretch",
                opacity=0.5,
                layer="below",
            )

    fig.update_layout(
        # clickmode='event+select',
        # width=700,
        height=700,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#272a31",
        plot_bgcolor="#272a31",
        showlegend=False
    )

    return fig
            
            
