import pandas as pd
import os
pd.options.mode.chained_assignment = None  # default='warn'

"""train_labels = pd.read_csv('../../jester-v1-train.csv', sep=';', names = ['Folder name', 'Gesture'])
#print(train_labels.head())

df_dict = {}
df_dict['Swiping Down'] = train_labels[train_labels.Gesture == 'Swiping Down']
df_dict['Swiping Up'] = train_labels[train_labels.Gesture == 'Swiping Up']
df_dict['Swiping Left'] = train_labels[train_labels.Gesture == 'Swiping Left']
df_dict['Swiping Right'] = train_labels[train_labels.Gesture == 'Swiping Right']
df_dict['Thumb Down'] = train_labels[train_labels.Gesture == 'Thumb Down']

df_dict['No gesture'] = train_labels[train_labels.Gesture == 'No gesture']
df_dict['Doing other things'] = train_labels[train_labels.Gesture == 'Doing other things']

#random_video = df_dict['Doing other things'].to_numpy()[0, 0]
for key in df_dict:
    print(key)
    ima = 0
    nema = 0
    for index, row in df_dict[key].iterrows():
        folder_name = row['Folder name']
        if not os.path.exists('../../20bn-jester-v2/'+str(folder_name)):
            nema += 1
            df_dict[key].drop(index, inplace=True)
        else:
            ima += 1
    print('Ima: ' + str(ima) + " Nema: " + str(nema))

for key in df_dict:
    df_dict[key].to_pickle(str(key)+'.df.pickle')"""

df_dict = {}
df_dict['Swiping Down'] = train_labels[train_labels.Gesture == 'Swiping Down']
df_dict['Swiping Up'] = train_labels[train_labels.Gesture == 'Swiping Up']
df_dict['Swiping Left'] = train_labels[train_labels.Gesture == 'Swiping Left']
df_dict['Swiping Right'] = train_labels[train_labels.Gesture == 'Swiping Right']
df_dict['Thumb Down'] = train_labels[train_labels.Gesture == 'Thumb Down']

df_dict['No gesture'] = train_labels[train_labels.Gesture == 'No gesture']
df_dict['Doing other things'] = train_labels[train_labels.Gesture == 'Doing other things']

"""img_array = []
for filename in os.listdir('../../20bn-jester-v2/'+str(754)):
    img = cv2.imread('../../20bn-jester-v2/'+str(754)+'/'+filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()"""