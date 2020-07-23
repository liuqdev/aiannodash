import numpy as np
from skimage.morphology import erosion, dilation
from skimage import measure
from sklearn.cluster import KMeans


# reference: https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
def kmeans_seg(img, zone=None):
    
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

    


    
# # https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
# # 目前只适用于单通道
# def kmeans_seg(img, display=False, save_to=None):
#     #Standardize the pixel values
#     row_size= img.shape[0]
#     col_size = img.shape[1]
    
#     mean = np.mean(img)
#     std = np.std(img)
#     img = img-mean
#     img = img/std
#     # Find the average pixel value near the lungs
#     # to renormalize washed out images
#     middle = img[int(col_size/8):int(col_size/8*7),int(row_size/8):int(row_size/8*7)] 
#     mean = np.mean(middle)  
#     max = np.max(img)
#     min = np.min(img)
#     # To improve threshold finding, I'm moving the 
#     # underflow and overflow on the pixel spectrum
#     img[img==max]=mean
#     img[img==min]=mean
    
#     #
#     # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
#     #
#     kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
#     centers = sorted(kmeans.cluster_centers_.flatten())
#     threshold = np.mean(centers)
#     thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

#     # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
#     # We don't want to accidentally clip the lung.

#     eroded = erosion(thresh_img,np.ones([3,3]))
#     dilated = dilation(eroded,np.ones([8, 8]))

#     labels = measure.label(dilated) # Different labels are displayed in different colors
    
#     label_vals = np.unique(labels)
#     regions = measure.regionprops(labels)
#     good_labels = []
#     for prop in regions:
#         B = prop.bbox
#         if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/10 and B[2]<col_size/10*9:
#             good_labels.append(prop.label)
#     mask = np.ndarray([row_size,col_size],dtype=np.int8)
#     mask[:] = 0

#     #
#     #  After just the lungs are left, we do another large dilation
#     #  in order to fill in and out the lung mask 
#     #
#     for N in good_labels:
#         mask = mask + np.where(labels==N,1,0)
#     # mask = dilation(mask, np.ones([10,10])) # one last dilation

#     if (display):
#         fig, ax = plt.subplots(3, 2, figsize=[12, 12])
#         ax[0, 0].set_title("Original")
#         ax[0, 0].imshow(img, cmap='gray')
#         ax[0, 0].axis('off')
#         ax[0, 1].set_title("Threshold")
#         ax[0, 1].imshow(thresh_img, cmap='gray')
#         ax[0, 1].axis('off')
#         ax[1, 0].set_title("After Erosion and Dilation")
#         ax[1, 0].imshow(dilated, cmap='gray')
#         ax[1, 0].axis('off')
#         ax[1, 1].set_title("Color Labels")
#         ax[1, 1].imshow(labels)
#         ax[1, 1].axis('off')
#         ax[2, 0].set_title("Final Mask")
#         ax[2, 0].imshow(mask, cmap='gray')
#         ax[2, 0].axis('off')
#         ax[2, 1].set_title("Apply Mask on Original")
#         ax[2, 1].imshow(mask*img, cmap='gray')
#         ax[2, 1].axis('off')
#         if save_to is not None:
#             plt.savefig(save_to, bbox_inches='tight')
#         plt.tight_layout()
#         plt.show()
#     return mask*img