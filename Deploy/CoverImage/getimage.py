import requests
import pandas as pd
import numpy as np
import shutil
def download_img(img_url,save_path):
    headers={'user - agent': 'Mozilla / 5.0(Windows NT 10.0;\
        Win64;\
        x64) AppleWebKit / 537.36(KHTML, like\
        Gecko) Chrome / 83.0\
        .4103\
        .116\
        Safari / 537.36'}
    r = requests.get(img_url, headers=headers, stream=True)
    code=r.status_code # 返回状态码
    if r.status_code == 200:
        open(save_path, 'wb').write(r.content) # 将内容写入图片
    del r
    return code

if __name__ == '__main__':
    # 下载要的图片
    movies=pd.read_csv('movies_mysql.csv',encoding='utf-8')
    movies=movies.values
    count=0
    imgcount=0
    #40052
    for i in range(40052):
        id=movies[i][0]
        url=movies[i][4]
        save_path=str(id)+'.jpg'
        if str(url)!='nan':           
            imgcount=imgcount+1
            img_url=url
            code=download_img(img_url,save_path)
            if code!=200:
                count=count+1
                shutil.copy("./img/default.jpg",save_path)

    #应该至少有1w6，如果灭有说明代码写错了
    print("有"+str(imgcount)+"个电影拥有封面")
    print("有"+str(count)+"个电影封面爬取失败")
    
        

    
    



