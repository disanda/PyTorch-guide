# save_image()
和make_grid()类似,不过可以直接保存tensor为图片，即输入是tensor格式:[N,C,W,H]

>torchvision.utils.save_image(tensor, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, ep, i + 1, len(train_loader)), nrow=10)


## refrence
https://mathpretty.com/11050.html
