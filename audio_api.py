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
import xpinyin
from xpinyin import Pinyin
import re

lfasr_host = 'https://raasr.xfyun.cn/v2/api'
# 请求的接口名
api_upload = '/upload'
api_get_result = '/getResult'
p = Pinyin()


def chinese_to_arabic(cn_num):
  cn_num_dict = {
      '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, 
      '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, 
      '时': 10, '两':2
  }

  return cn_num_dict.get(cn_num, 0) 

physical_quantities = {
  "前缘半径": ["前沿半径", "前缘半径", "边缘半径","前圆半径","前额半径","迁移半径"],
  "上表面峰值": ["上表面风值","上等的峰值","受表面峰值","上半年峰值","尚表面峰值"],
  "下表面峰值": ["下表面风值", "底部峰值"],
  "后缘角": ["后元角", "后圆角", "后源角", "后原角", "后远角", "后员角","后眼角"]
}

def check(candidate,prompt_pinyin):
    i = 0
    for c in prompt_pinyin:
        if i<len(candidate) and c==candidate[i]:
            i+=1
    return i/len(candidate)
          
        

def process_pingyin(prompt_pinyin):
    # 判断
    a = check('qian-yuan-bang-jing',prompt_pinyin)
    b = check('shang-biao-mian-feng-zhi',prompt_pinyin)
    c = check('xia-biao-mian-feng-zhi',prompt_pinyin)
    d = check('hou-yuan-jiao',prompt_pinyin)
    ## 找到在s中匹配最成功的率
    cur = max(a,b,c,d)
    if a==cur:
        return '前缘半径'
    elif b==cur:
        return '上表面峰值'
    elif c==cur:
        return '下表面峰值'
    else:
        return '后缘角'

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
                      # print(arabic_num, operation, correct)
                      # print(operations[operation] == "+")
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
        data = open(upload_file_path, 'rb').read(file_len)
        response = requests.post(url =lfasr_host + api_upload+"?"+urllib.parse.urlencode(param_dict),
                                headers = {"Content-type":"application/json"},data=data)
        result = json.loads(response.text)
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
        status = 3
        # 建议使用回调的方式查询结果，查询接口有请求频率限制
        while status == 3:
            response = requests.post(url=lfasr_host + api_get_result + "?" + urllib.parse.urlencode(param_dict),
                                     headers={"Content-type": "application/json"})
            result = json.loads(response.text)
            status = result['content']['orderInfo']['status']
            if status == 4:
                break
            time.sleep(5)        
        return result



def audio2parsec(upload_file_path):
  t1 = time.time()
  api = RequestApi(appid="2c481fcb",
                  secret_key="40007fc72d491a02b571343aa4e80de9",
                  upload_file_path=upload_file_path)
  t2 = time.time() - t1

  result = api.get_result()
  ss = json.loads(result['content']['orderResult'])['lattice']
  prompt = ''
  equal = ''
  for s in ss:
      s = json.loads(s['json_1best'])['st']['rt'][0]['ws']
      for i in s:
          prompt += i['cw'][0]['w']

  print(prompt)
  results = process_text(prompt)
  for key, value in results.items():
      if value != 0:
        return key,value
  print(results)
  ## 写一个逻辑，能够将prompt和 physical_quantities匹配上
  prompt_pingyin = p.get_pinyin(prompt)
  key = process_pingyin(prompt_pingyin)
  num = re.findall(r'\d+|[一二三四五六七八九十时两]+', prompt)
  try:
      arabic_num = int(num[0])
  except ValueError:
      arabic_num = chinese_to_arabic(num[0])
  results[key] = arabic_num
  for key, value in results.items():
      if value != 0:
        return key,value
  print(results)
  return prompt,-1