import numpy as np
import time,datetime
import random
import pickle

import pymysql


def Normalization_gowalla(inX):
    return 1.0 / (1 + np.exp(-inX))

def Normalization_epinions(x,Max=5.0,Min=0):
    x = (x - Min) / (Max - Min)
    return x

class Dataset(object):
    def __init__(
            self,
            path='data/',
            negative=5,
            train_step = 11,
            u_counter = 4630,
            s_counter = 26991,
            data_name='epinions'):
        '''
            Constructor:
                data_name: Name of Dataset
        '''
        self.data_name = data_name
        self.train_id_list = []
        self.train_rating_list = []
        self.test_id_list = []
        self.test_rating_list = []
        self.train_data = {}
        self.test_data = {}
        self.user_dict = {}
        self.spot_dict = {}
        self.user_enum = {}
        self.spot_enum = {}
        self.links = {}
        self.links_array = {}
        self.u_counter = u_counter
        self.s_counter = s_counter
        self.path = path
        self.negative = negative
        self.train_step = train_step

    #
    def generate(self):

        self.get_inter_data()
        self.get_train_data()
        self.get_test_rating()
        self.get_test_link()

    def get_inter_data(self):
        user_r = {}
        count = 0
        f = open("data/"+self.data_name+".train.rating")  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法
        while line:
            line = line.rstrip()
            arr = line.split('\t')
            user_id = int(arr[0])
            spot_id = int(arr[1])

            if user_id in user_r.keys():
                if spot_id not in user_r[user_id]:
                    self.train_id_list.append([user_id, spot_id])
                    count += 1
                    user_r[user_id].append(spot_id)
            else:
                self.train_id_list.append([user_id, spot_id])
                count += 1
                user_r[user_id]= [spot_id]

            line = f.readline()
        f.close()

        f = open("data/"+self.data_name+".train.link")  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法
        while line:
            line = line.rstrip()
            arr = line.split('\t')
            user1 = int(arr[0])
            if user1 not in user_r.keys():
                user_r[user1] = [1]
                for i in range(20):
                    self.train_id_list.append([user1, i+1])
            line = f.readline()  # 调用文件的 readline()方法
        f.close()

        f = open("data/"+self.data_name+".train.rating")  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法
        train_data_num_T = 0
        while line:
            line = line.rstrip()
            arr = line.split('\t')
            user_id = int(arr[0])
            spot_id = int(arr[1])
            rating = float(arr[2])
            if self.data_name == 'gowalla':
                rating = Normalization_gowalla(rating)
            else:
                rating = Normalization_epinions(rating)
            step = int(arr[3])
            if user_id in self.user_dict.keys():
                self.user_dict[user_id].append([spot_id,rating, step])
            else:
                self.user_dict[user_id] = [[spot_id,rating, step]]

            if spot_id in self.spot_dict.keys():
                self.spot_dict[spot_id].append([user_id,rating, step])
            else:
                self.spot_dict[spot_id] = [[user_id,rating, step]]

            train_data_num_T += 1
            line = f.readline()  # 调用文件的 readline()方法
        f.close()


        f = open("data/"+self.data_name+".train.link")  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法
        while line:
            line = line.rstrip()
            arr = line.split('\t')
            user1 = int(arr[0])
            user2 = int(arr[1])
            step = int(arr[2])

            if user1 not in self.links.keys():
                self.links[user1] = [[user2, step]]
            else:
                self.links[user1].append([user2, step])
            if user1 not in self.links_array.keys():
                self.links_array[user1] = [user2]
            else:
                self.links_array[user1].append(user2)
            line = f.readline()
        f.close()


        inter_data = {}
        inter_data['ids'] = self.train_id_list
        inter_data['user_dict'] = self.user_dict
        inter_data['spot_dict'] = self.spot_dict
        inter_data['links'] = self.links
        inter_data['links_array'] = self.links_array

        with open("data/inter_"+self.data_name+".pkl", 'wb') as f:
            pickle.dump(inter_data, f)

    print("finish getting inter_data...")

    def get_train_data_by_id(self, id=0):
        res = {}
        rating_indicator = np.zeros(self.train_step, dtype='int32')
        rating_pre = np.zeros(self.train_step, dtype='float32')
        spot_attr = []
        link_res = []
        link_predict_weight = []

        if self.train_id_list[id][0]  in self.user_dict.keys():
            for record in self.user_dict[self.train_id_list[id][0]]:
                if record[0] == self.train_id_list[id][1]:
                    rating_pre[record[2]] = record[1]
                    rating_indicator[record[2]] = 1

        if self.train_id_list[id][1] in self.spot_dict.keys():
            for record in self.spot_dict[self.train_id_list[id][1]]:
                spot_attr.append([record[0], record[1], record[2]])

        if self.train_id_list[id][0] in self.links.keys():
            for record in self.links[self.train_id_list[id][0]]:
                link_res.append([record[0], record[1]])
                link_predict_weight.append([record[0], record[1], 1.0])
                for i in range(self.negative):
                    link_predict_weight.append([random.randrange(start=1,stop=self.u_counter),record[1] ,1.0/self.negative])


        res['user_id'] = self.train_id_list[id][0]
        res['spot_id'] = self.train_id_list[id][1]
        res['spot_attr'] = spot_attr
        res['rating_pre'] = rating_pre
        res['rating_indicator'] = rating_indicator
        res['link_res'] = link_res
        res['link_predict_weight'] = link_predict_weight
        return res
    def get_train_data(self):
        print("start getting train_data...")
        res_t = {}
        for i in range(len(self.train_id_list)):
            res_one = self.get_train_data_by_id(id = i)
            res_t[i] = res_one
        print("finish getting train_data...")
        with open("data/train_"+self.data_name+".pkl", 'wb') as f:
            pickle.dump(res_t, f)

    def get_test_rating(self):
        print("start getting test rating...")
        f = open("data/"+self.data_name+".test.rating")  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法
        while line:
            line = line.rstrip()
            arr = line.split('\t')
            user_id = int(arr[0])
            spot_id = int(arr[1])
            rating = float(arr[2])
            if self.data_name == 'gowalla':
                rating = Normalization_gowalla(rating)
            else:
                rating = Normalization_epinions(rating)

            if [user_id, spot_id] not in self.test_id_list:
                self.test_id_list.append([user_id, spot_id])
                self.test_rating_list.append(rating)

            line = f.readline()
        f.close()

        test_data = {}
        test_data['ids'] = self.test_id_list
        test_data['rating'] = self.test_rating_list
        test_data['user_dict'] = self.user_dict
        test_data['spot_dict'] = self.spot_dict
        test_data['links'] = self.links
        print("finish getting test rating...")
        with open("data/test_rating_"+self.data_name+".pkl", 'wb') as f:
            pickle.dump(test_data, f)
    def get_test_link(self):
        self.last_pre = {}
        self.till_record = {}
        print("start getting test link...")
        f = open("data/"+self.data_name+".test.link")  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法
        while line:
            line = line.rstrip()
            arr = line.split('\t')
            user1 = int(arr[0])
            user2 = int(arr[1])

            if user1 not in self.last_pre.keys():
                self.last_pre[user1] = [user2]
            else:
                self.last_pre[user1].append(user2)
            if user1 not in self.till_record.keys():
                if user1 in self.links_array.keys():
                    self.till_record[user1] = self.links_array[user1]
                else:
                    self.till_record[user1] = []
            line = f.readline()
        f.close()

        test_link = {}
        test_link['last_pre'] = self.last_pre
        test_link['till_record'] = self.till_record
        with open("data/test_link_"+self.data_name+".pkl", 'wb') as f:
            pickle.dump(test_link, f)
        print("finish getting test link...")


