import csv
import os, sys
import pickle
import numpy as np


class batch_gen:
    def __init__(self, csv_path):
        path, ext = os.path.splitext(csv_path)
        name, ext = os.path.splitext(os.path.basename(csv_path))
        path = path.replace(name, "")
        self.name = name
        self.csv_path = csv_path
        self.data_pickle_name = name + ".pickle"
        self.data_path = path
        self.data_pickle_path = os.path.join(self.data_path, self.data_pickle_name)
        self.data = []
        print("batch_gen",self.name)
        self.numpy_data_prepare()
        self.length, self.dim = self.get_lengths()
        print("batch_gen", "length = " + str(self.length), "dim = " + str(self.dim))
    
    def get_train_batch(self, index, batch_size, input_size, dist, output_size):
        self.max_index = self.length - (batch_size + input_size + dist + output_size)
        self.u_index = index
        if not self.max_index > self.u_index:
            self.u_index = index % self.max_index
        self.in_data = []
        self.out_data = []
        for i in range(batch_size):
            self.in_data.append(self.data[self.u_index + i : self.u_index + input_size + i])
            self.out_data.append(self.data[self.u_index + dist + i : self.u_index + dist + output_size + i])
        return np.array(self.in_data, dtype=np.float32), np.array(self.out_data, dtype=np.float32)

    def get_lengths(self):
        return len(self.data), len(self.data[0])

    def numpy_data_prepare(self):
        if os.path.isfile(self.data_pickle_path):
            self.restore_pickle_file()
            print("batch_gen","exist pickle file", self.data_pickle_path)
        else:
            self.make_pickle_file()
            print("batch_gen","load and make pickle file", self.data_pickle_path)

    def restore_pickle_file(self):
        with open(self.data_pickle_path, mode='rb') as f:
            self.data = pickle.load(f)

    def make_pickle_file(self):
        raw_data = self.read_csv(self.csv_path)
        self.data = []
        for r in raw_data:
            self.data.append(np.array(r, dtype=np.float32))
        self.data = np.array(self.data, dtype=np.float32)
        with open(self.data_pickle_path, mode='wb') as f:
            pickle.dump(self.data, f)

    def read_csv(self, _path):
        l = []
        with open(_path) as f:
	        reader = csv.reader(f)
	        for row in reader:
	            l.append(row)
        return l

class kinect_data:
    def __init__(self, csv_path, scale = 1.0):
        self.origin_csv_path = csv_path
        self.scale = scale
        self.raw_data = self.read_csv(self.origin_csv_path)
        self.shaped_data = []
        self.shaped_data_path = ""
        self.shaping_data()

    def create_batch_gen(self):
        return batch_gen(self.shaped_data_path)

    def reshape_overlap(self, data):
        use_index = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47] 
        r_data = []
        for r in data:
            r_r = []
            for u_i in use_index:
                r_r.extend([r[u_i * 3 + 0], r[u_i * 3 + 1], r[u_i * 3 + 2]])
            r_data.append(r_r)
        return r_data

    def shaping_data(self):
        self.shaped_data_path = self.origin_csv_path.split(".")[0] + "_shape_scale_" + str(int(1 / self.scale)) + ".csv"
        if os.path.isfile(self.shaped_data_path):
            print("kinect_data", "exist shaped data csv")
            self.read_shaped_data(self.shaped_data_path)
        else:
            print("kinect_data", "make shaped data csv")
            self.make_shaped_data(self.shaped_data_path)

    def read_shaped_data(self, shaped_data_path):
        self.shaped_data = self.read_csv(shaped_data_path)

    def make_shaped_data(self, shaped_data_path):
        shaped = []
        for r in self.raw_data:
            raw = []
            for c in r:
                raw.append(float(c) * self.scale)
            shaped.append(raw)
        if len(shaped[0]) == 144:
            shaped = self.reshape_overlap(shaped)
        self.shaped_data = shaped
        self.write_csv(self.shaped_data, shaped_data_path)

    def read_csv(self, _path):
        l = []
        with open(_path) as f:
	        reader = csv.reader(f)
	        for row in reader:
	            l.append(row)
        return l	
	
    def write_csv(self, __data, __path):
	    with open(__path, 'w') as c:
		    writer = csv.writer(c, lineterminator='\n')
		    writer.writerows(__data)

