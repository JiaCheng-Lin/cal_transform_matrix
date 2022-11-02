# cal_transform_matrix

## **The steps of collecting data and cal the Transform T :**
### 1.  execute the following script. 
```bash
python collect_data.py
```
Contains the coordinates of the **point** of the person in **mmwave** and the coordinates of the **corresponding point** in the **webcam**

#### **Note:** Please keep only one person in the webcam/mmwave view. Otherwise the Transform T will be inaccurate.
### 
### 2. And then execute the following script. 
```bash
python cal_T.py
```
or
```bash
python cal_T_regression.py
```

"cal_T_regression.py" using polynomial regression to project mmwave pts to img.

#### **Note:** you need to change the 'data_path' to your new .npy file name

After get the transform **T**, update it to the .../inference/mmwave_utils.mmwave.py -> **process_mmwave()** function.
