[[_TOC_]]

## DataRead

### `train = DataRead(folder_path, data_type, batch_size, batch_num, shuffle=False, seed=False)` <br>
Load in the needed pictures (original images, cropped images, crops), and .csv file that contains the crop coordinates <br>
- `folder_path`: (str) This folder that contains the needed folders and files <br> 
- `data_type`: (str) 'train','valid' or 'test' determining from which folder should we get the data <br>
- `batch_size`: (int) the batch size for the data <br>
- `batch_num`: (int) the number of batches for the data <br>
- `shuffle`: (bool) if true the order of the data will be suffled <br>
- `seed`: (bool) if true the shuffle will have a random seed <br>

Important variables in the DataStream class: <br>
- `train.csv`: an array that contains the place of the crop on the picture. <br>
- `train.target_images`: an array that contains the original pictures <br>       
- `train.crop_images`: an array that contains the crops <br>
- `train.cropped_images`: an array that contains the cropped images <br> 
- `train.batch_size`: integer that shows the batch size <br>
- `train.batch_num`: integes that shows the number of batches <br>

-------
### `train.reset()` <br>
puts new data into the object <br>

-------
### `change_batch_size(new_batch_size)` <br>
changes the batch size in the object <br>
- `new_batch_size`: (int) the new batch size <br>

-------
### `change_batch_num(new_batch_num)` <br>
changes the number of batches in the object <br>
- `new_batch_num`: (int) the new number if batches <br>

