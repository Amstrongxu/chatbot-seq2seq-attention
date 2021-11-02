# _*_ coding:utf-8 _*_
# @Time : 2021/10/16 19:04
# @Author : xupeng
# @File : getConfig.py
# @software : PyCharm

from configparser import ConfigParser

#解析ini配置文件，返回字典，键值对信息
def get_config(config_file='./seq2seq.ini'):
    # print(config_file)
    parser = ConfigParser()
    parser.read(config_file,encoding='utf-8')
    # get the ints, floats and strings
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    # _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _confg_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _confg_strings)

# res_dict = get_config()
# print(res_dict)
