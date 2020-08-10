from threading import Thread
import collections
import cv2
from torchvision.transforms import *
import copy
import time
from PIL import Image
from resnext101 import load
import os
from torch.autograd import Variable
import torch
from spatial_transforms import *

switcher = {
                    0: "other things",
                    1: "no gesture",
                    2: "down",
                    3: "left",
                    4: "right",
                    5: "up",
                    6: "thumb down"
                }

tested_sequences = []
pillow_sequence = collections.deque(16*[0], 16)
predictions = []
frame_stream = collections.deque(16*[0], 16)
spatial_transform = Compose([
        # Scale(opt.sample_size),
        Scale(112),
        CenterCrop(112),
        ToTensor(1),
        Normalize([0, 0, 0], [1, 1, 1])
        ])

model = load()
model.eval()

stream = cv2.VideoCapture('dugacak_test.webm')
ret = True
i = 0
while ret and stream.isOpened():
    ret, frame = stream.read()
    if ret:
        i += 1
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        aug_frame = cv2.resize(frame, (320, 240))
        aug_frame = Image.fromarray(cv2.cvtColor(aug_frame, cv2.COLOR_BGR2RGB))
        pillow_sequence.append(aug_frame)
        aug_frame = aug_frame.convert('RGB')
        aug_frame = spatial_transform(aug_frame)
        frame_stream.append(aug_frame)
        if i == 16:
            start = time.time()
            tested_sequences.append(copy.deepcopy(pillow_sequence))
            print('Copy time: ' +str(time.time() - start))
            i = 0
            im_dim = frame_stream[0].size()[-2:]
            try:
                test_data = torch.cat(list(frame_stream), 0).view((16, -1) + im_dim).permute(1, 0, 2, 3)
            except Exception as e:
                print('Greska')
                pass
            inputs = torch.cat([test_data], 0).view(1, 3, 16, 112, 112)
            inputs = Variable(inputs)
            inputs = inputs[:, :, :, :, :]
            inputs = torch.Tensor(inputs.numpy()[:, :, ::1, :, :])
            # model
            start = time.time()
            outputs = model(inputs)
            outputs = outputs[:, [0, 2, 15, 16, 17, 18, 19]].argmax(1)
            predictions.append(outputs[0].item())
            print(outputs)
            print('Time = ' + str(time.time() - start))
stream.release()
cv2.destroyAllWindows()
for i, test_seq in enumerate(tested_sequences):
    if not os.path.exists(str(i+1)):
        os.makedirs(str(i+1))
    for j, img in enumerate(list(test_seq)):
        img.save(str(i+1)+'/'+str(j+1)+'.jpg')
for i, pred in enumerate(predictions):
    print(str(i+1) + ' ' + switcher[pred])
