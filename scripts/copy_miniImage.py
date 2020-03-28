import os
local_image_root = '/Users/liweijie/datasets/images'
notsend_image_root = '/Users/liweijie/datasets/havenotsend/.'
fname_list = os.listdir(local_image_root)
fname_list.sort()
have_copied = open('have_copied.txt','r').readlines()
have_copied = [line.rstrip('\n') for line in have_copied]
for i,fname in enumerate(fname_list):
    print('%d/%d'%(i,len(fname_list)))
    if not fname in have_copied:
        fpath = os.path.join(local_image_root,fname)
        cmd = 'cp ' + fpath + ' ' + notsend_image_root
        os.system(cmd)
    