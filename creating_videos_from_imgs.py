import cv2
import pandas as pd
import os

df_dict = {}
df_dict['Swiping_Down'] = pd.read_pickle("Swiping Down.df.pickle")
df_dict['Swiping_Up'] = pd.read_pickle("Swiping Up.df.pickle")
df_dict['Swiping_Left'] = pd.read_pickle("Swiping Left.df.pickle")
df_dict['Swiping_Right'] = pd.read_pickle("Swiping Right.df.pickle")
df_dict['Thumb_Down'] = pd.read_pickle("Thumb Down.df.pickle")

df_dict['No_gesture'] = pd.read_pickle("No gesture.df.pickle")
df_dict['Doing_other_things'] = pd.read_pickle("Doing other things.df.pickle")

for key in df_dict:
    print(key)
    for index, row in df_dict[key].iterrows():
        folder_name = row['Folder name']
        img_array = []
        for filename in os.listdir('../20bn-jester-v2/'+str(folder_name)):
            img = cv2.imread('../20bn-jester-v2/'+str(folder_name)+'/'+filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

        out = cv2.VideoWriter('../videos/' + str(key) + '/' + str(folder_name) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()