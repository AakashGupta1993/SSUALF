>>> import numpy as np
>>> data_dir = './data'
>>> import os.path
>>> import scipy.misc
>>> import helper
>>> image_shape = (160, 576)
>>> get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
>>> x, y = next(get_batches_fn)
Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      TypeError: 'function' object is not an iterator
      >>> x, y = next(get_batches_fn(10))
      >>> x[0].shape
      (160, 576, 3)
      >>> y[0].shape
      (160, 576, 2)
      >>> y[0].min()
      False
      >>> y[0].max()
      True
      >>> y[0][:,:,0].max()
      False
      >>> y[0][:,:,0].min()
      False
      >>> y[0][:,:,1].min()
      True
      >>> y[0][:,:,1].max()
      True
      >>> gt_image = scipy.misc.imresize(scipy.misc.imread("data/d"), image_shape)
      data_dir  def       del       delattr(  dict(     dir(      divmod(
          >>> gt_image = scipy.misc.imresize(scipy.misc.imread("data/d"), image_shape)
          data_dir  def       del       delattr(  dict(     dir(      divmod(
              >>> gt_image = scipy.misc.imresize(scipy.misc.imread("data/data_road/training/gt_image_2/um_road_000000.jpg"), image_shape)
              >>> gt_image.shape
              (160, 576, 3)
              >>> gt_image[:,:,0].shape
              (160, 576)
              >>> gt_image[:,:,0].min()
              249
              >>> gt_image[:,:,0].max()
              255
              >>> gt_image[:,:,2].max()
              255
              >>> gt_image[:,:,2].min()
              0
              >>> y[:,:,2].min()
              False
