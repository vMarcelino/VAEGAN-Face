from tensorflow.keras.utils import Sequence
import os
import numpy as np
import face


class DataGenerator(Sequence):
    #class DataGenerator():
    def __init__(self,
                 folder,
                 batch_size,
                 test=None,
                 split=0.7,
                 output_shape=(128, 128),
                 max_samples=0,
                 max_samples_is_percentage=False):

        self.files = [os.path.join(folder, f) for f in os.listdir(folder)]
        if max_samples:
            if max_samples_is_percentage:
                self.files = self.files[:len(self.files) * max_samples]
            else:
                self.files = self.files[:max_samples]

        self.output_shape = output_shape
        if test is not None:
            sz = len(self.files)

            if test:
                self.files = self.files[int(sz * split):]  # test: from split point to end
            else:
                self.files = self.files[:int(sz * split)]  # train: from start to split point

        self.batch_size = batch_size

    def __len__(self):
        l = np.ceil(len(self.files) / self.batch_size)
        return int(l)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        if end > len(self.files):
            end = len(self.files)

        filenames = [self.files[k] for k in range(start, end)]

        # extract face on 128x128x3 shape

        file_batch = []
        for filename in filenames:
            try:
                file_batch.append(face.get_face(filename, resize=self.output_shape))
            except:
                pass

        file_batch = np.array(file_batch, dtype=float)
        file_batch /= 255

        return file_batch


if __name__ == "__main__":
    g = DataGenerator('data', 2**10, (3, 3))
    processes = []
    import multiprocessing
    for i in range(len(g)):
        # g.__getitem__(i)
        p = multiprocessing.Process(target=g.__getitem__, args=(i, ))
        processes.append(p)
        p.start()

    for i, p in enumerate(processes):
        p.join()
        print(i, 'terminou')
