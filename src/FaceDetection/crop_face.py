
def crop_face(image):
    """takes as input an rgb image of a centered face. crops the skin part out using the HSV distribution of the center
    part. Fills in the holes and returns the cropped image as a grey-value matrix"""
    assert shape(image)==(300,300,3)
    subimage = image[110:190,110:190]

    #figure(figsize=(10,5))

    #subplot(3,2,1)
    #imshow_rgb(image)
    #subplot(3,2,2)
    #imshow_rgb(subimage)

    subimage_hsv=cv2.cvtColor(subimage, cv2.COLOR_RGB2HSV)
    image_hsv=cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h,w,colors=shape(subimage)
    HSV=zeros([h*w,3])

    #subplot(3,2,3)
    for i in range(3):
        HSV[:,i]=ravel(subimage_hsv[:,:,i])
    #hist(HSV,label=['H','S','V']);
    #legend();

    Mean_hsv=mean(HSV,axis=0)
    STD_hsv=std(HSV,axis=0)

    #print 'mean=',Mean_hsv,', std=', STD_hsv

    notface=sum(((image_hsv-Mean_hsv)/STD_hsv)**2,axis=2)
    mask=(1*(notface<10)).astype(int)
    #subplot(3,2,4)
    #imshow(mask);

    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask=label_im==argmax(sizes)
    #imshow(mask)

    mask2=binary_fill_holes(mask)
    #subplot(3,2,5)
    #imshow(mask2)

    image2=copy(image)
    for i in range(3):
        image2[:,:,i] *= mask2   #2.astype(int64)

    grey_image=sum(image2,axis=2)
    subplot(3,2,6)
    #imshow(grey_image, cmap='Greys_r')
    return grey_image