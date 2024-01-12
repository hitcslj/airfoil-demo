# -*- coding: utf-8 -*-
import base64
import hashlib
import hmac
import json
import os
import time
import requests
import urllib
import random

lfasr_host = 'https://raasr.xfyun.cn/v2/api'
# 请求的接口名
api_upload = '/upload'
api_get_result = '/getResult'
import re

def chinese_to_arabic(cn_num):
  cn_num_dict = {
      '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, 
      '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, 
      '时': 10, '两':2
  }

  return cn_num_dict.get(cn_num, 0) 

physical_quantities = {
  "前缘半径": ["前沿半径", "前缘半径", "边缘半径","前圆半径","前额半径"],
  "上表面峰值": ["上表面风值","上等的峰值","受表面峰值","上半年峰值"],
  "下表面峰值": ["下表面风值", "底部峰值"],
  "后缘角": ["后元角", "后圆角", "后源角", "后原角", "后远角", "后员角","后眼角"]
}

def process_text(text):
  results = {
      "前缘半径": 0,
      "上表面峰值": 0,
      "下表面峰值": 0,
      "后缘角": 0
  }
  operations = {
      "加": "+",
      "减": "-",
  }
  for correct, errors in physical_quantities.items():
      if correct in text or any(error in text for error in errors):
          for operation in operations:
              if operation in text:
                  num = re.findall(r'\d+|[一二三四五六七八九十时两]+', text)
                  if num:
                      try:
                          arabic_num = int(num[0])
                      except ValueError:
                          arabic_num = chinese_to_arabic(num[0])
                      #operation = operations[operation]
                      print(arabic_num, operation, correct)
                      print(operations[operation] == "+")
                      if operations[operation] == "+":
                          results[correct] += arabic_num
                      else:
                          results[correct] -= arabic_num

  return results


class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path
        self.ts = str(int(time.time()))
        self.signa = self.get_signa()

    def get_signa(self):
        appid = self.appid
        secret_key = self.secret_key
        m2 = hashlib.md5()
        m2.update((appid + self.ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        return signa


    def upload(self):
        print("上传部分：")
        upload_file_path = self.upload_file_path
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)

        param_dict = {}
        param_dict['appId'] = self.appid
        param_dict['signa'] = self.signa
        param_dict['ts'] = self.ts
        param_dict["fileSize"] = file_len
        param_dict["fileName"] = file_name
        param_dict["duration"] = "200"
        print("upload参数：", param_dict)
        data = open(upload_file_path, 'rb').read(file_len)
        print(data)
        response = requests.post(url =lfasr_host + api_upload+"?"+urllib.parse.urlencode(param_dict),
                                headers = {"Content-type":"application/json"},data=data)
        print("upload_url:",response.request.url)
        result = json.loads(response.text)
        print("upload resp:", result)
        return result


    def get_result(self):
        uploadresp = self.upload()
        orderId = uploadresp['content']['orderId']
        param_dict = {}
        param_dict['appId'] = self.appid
        param_dict['signa'] = self.signa
        param_dict['ts'] = self.ts
        param_dict['orderId'] = orderId
        param_dict['resultType'] = "transfer,predict"
        print("")
        print("查询部分：")
        print("get result参数：", param_dict)
        status = 3
        # 建议使用回调的方式查询结果，查询接口有请求频率限制
        while status == 3:
            response = requests.post(url=lfasr_host + api_get_result + "?" + urllib.parse.urlencode(param_dict),
                                     headers={"Content-type": "application/json"})
            # print("get_result_url:",response.request.url)
            result = json.loads(response.text)
            # print(result)
            status = result['content']['orderInfo']['status']
            print("status=",status)
            if status == 4:
                break
            time.sleep(5)        
        print("get_result resp:",result)
        return result



def audio2parsec(upload_file_path):
  api = RequestApi(appid="74d744fa",
                  secret_key="752071b4b90c4406a05ad3bc78b100e7",
                  upload_file_path=upload_file_path)

  result = api.get_result()
  ss = json.loads(result['content']['orderResult'])['lattice']
  prompt = ''
  equal = ''
  for s in ss:
      s = json.loads(s['json_1best'])['st']['rt'][0]['ws']
      for i in s:
          prompt += i['cw'][0]['w']
  # print(prompt) # 目前prompt识别出来的文字非常糟糕，接口暂时是写死的，后续可以考虑用别人的语音接口+字符串匹配规则


  print(prompt)
  results = process_text(prompt)
  for key, value in results.items():
      if value != 0:
        return key,value
  return '你好，可以说的再清楚一些么?',-1
  


  # def check(ss):
  #     for s in ss:
  #         if s in prompt:
  #             return s[0]
  #     return False

  # def prompt2param():
  #   a = ['前缘半径','前沿半径']
  #   b = ['上表面峰值']
  #   c = ['下表面峰值']
  #   d = ['后缘角','后圆角']
  #   if check(a):return check(a)
  #   if check(b):return check(b)
  #   if check(c):return check(c)
  #   if check(d):return check(d)
  #   return '你好，可以说的再清楚一些么？'
  
  # def prompt2value(prompt):
  #   one = ['1','一']
  #   two = ['2','']
  #   three = ['3',]
  #   four = []
  #   five = []
  #   six = []
  #   seven = []
  #   eight = []
  #   nine = []
  #   ten = []
  #   values = [1,2,3,4,5,6,7,8,9,10]
  #   chinese_values = ['一','二','三','四','五','六','七','八','九','十']

  #   for v in values:
  #       if str(v) in prompt:
  #           return v
  #   for v in chinese_values:
  #       if v in prompt:
  #           return chinese_values.index(v)+1
  #   return 0
  # param = prompt2param() # prompt 和 keys算相似度，取出来
  # value = prompt2value() # prompt 和 values算相似度，取出来
  # return param,value
    