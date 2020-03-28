import os
image_root = '/home/liweijie/Data/miniImagenet/images'
fname_list = os.listdir(image_root)
fname_list.sort()
account_file = open('have_copied.txt','w')
for fname in fname_list:
    account_file.write(fname+'\n')
