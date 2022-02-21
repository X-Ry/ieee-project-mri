
#This is old stuff i was trying out from week 1, ignore this lmao

import matplotlib.pyplot as plt

#BASE_IMG_PATH=os.path.join('C:\Users\great\PycharmProjects\neuroimaging')


try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')

test_image=nib.load(r'C:\Users\great\PycharmProjects\neuroimaging\mean_all_studies.nii').get_data()
test_image2=nib.load(r'C:\Users\great\PycharmProjects\neuroimaging\10OurStudy.nii').get_data()
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
ax1.imshow(test_image[test_image.shape[0]//2])
ax1.set_title('Image')
ax1.imshow(test_image2[test_image2.shape[0]//2])
ax1.set_title('Image2')

#test_image is a memmap,

from skimage.util import montage as montage2d
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(montage2d(test_image), cmap ='bone')
fig.savefig('ct_scan.png')

fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(montage2d(test_image2), cmap ='bone')
fig.savefig('ct_scan2.png')


#image_data = array_img.get_fdata()

print("finished")