from threading import Thread
import collections
import cv2
from torchvision.transforms import *
import time
from PIL import Image
from resnext101 import load
import os
from torch.autograd import Variable
import torch
from spatial_transforms import *


frame_stream = collections.deque(16*[0], 16)
spatial_transform = Compose([
        # Scale(opt.sample_size),
        Scale(112),
        CenterCrop(112),
        ToTensor(1),
        Normalize([0, 0, 0], [1, 1, 1])
        ])


def test_video_capture():
    stream = cv2.VideoCapture('nulti_test.webm')
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
            aug_frame = aug_frame.convert('RGB')
            aug_frame = spatial_transform(aug_frame)
            frame_stream.append(aug_frame)
    # Release everything if job is finished
    stream.release()
    cv2.destroyAllWindows()
    print(i)


def video_capture_thread():
    #for video in os.listdir('Swiping_Up'):
        #print('Swiping_Up/' + video)
        stream = cv2.VideoCapture(0)
        ret = True
        while ret and stream.isOpened():
            ret, frame = stream.read()
            if ret:
                cv2.imshow('frame', frame)
                time.sleep(0.07)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                aug_frame = cv2.resize(frame, (320, 240))
                aug_frame = Image.fromarray(cv2.cvtColor(aug_frame, cv2.COLOR_BGR2RGB))
                aug_frame = aug_frame.convert('RGB')
                aug_frame = spatial_transform(aug_frame)
                frame_stream.append(aug_frame)
        # Release everything if job is finished
        stream.release()
        cv2.destroyAllWindows()

try:
    compare_tensor = torch.load('input_tensor.pt', map_location=torch.device('cpu'))
    compare_tensor = compare_tensor.transpose(1, 2)
    """for a in compare_tensor:
        for i, img in enumerate(a):
            img = Image.fromarray(img.transpose(0, 1).transpose(1, 2).numpy().astype(np.uint8))
            img.save(os.path.join('0000/0000' + str(i+1) + ".jpg"))"""
    test_video_capture()
    print('Initializing...')
    model = load()
    model.eval()
    """thread = Thread(target=video_capture_thread)
        thread.setDaemon(True)
        thread.start()"""
    time.sleep(4)
    print('Ready for use')
    while True:
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
        print(inputs)
        #model
        start = time.time()
        outputs = model(inputs)
        print(outputs)
        outputs = outputs[:, [0, 2, 15, 16, 17, 18, 19]].argmax(1)
        print(outputs)
        print('Time = ' + str(time.time() - start))
except KeyboardInterrupt:
    pass
