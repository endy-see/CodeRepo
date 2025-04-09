import requests
from time import sleep
import pandas as pd
import os
import time
import datetime
import random
import json
import base64
from PIL import Image
from io import BytesIO
import imageio.v3 as iio


# # url = 'https://fund.eastmoney.com/data/fundranking.html#tall;c0;r;s1nzf;pn50;ddesc;qsd20240326;qed20250326;qdii;zq;gg;gzbd;gzfs;bbzt;sfbb'
# # 小红书
# # url = 'https://edith.xiaohongshu.com/api/sns/web/v2/comment/page?note_id=67cc3cb3000000000603d800&cursor=&top_comment_id=&image_formats=jpg,webp,avif&xsec_token=ABCCFz1mSimfjrgudHfNBTWVF9cvLliboQlx42uakdNSs%3D'

# pip install imageio pillow
def download_image(url, save_path):
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type')     # 默认格式是image/webp，需要转成jpg才能本地查看
            if 'image' in content_type:
                # webp_image = imageio.imread(BytesIO(response.content))
                webp_image = iio.imread(BytesIO(response.content))
                pil_image = Image.fromarray(webp_image)
                pil_image.save(save_path, 'JPEG')
                # print(f'Image successfully downloaded and saved to {save_path}')
            else:
                print(f'The URL does not point to an image, content type: {content_type}')
        else:
            print(f'Failed to download image. HTTP Status Code: {response.status_code}')
    except Exception as e:
        print(f'An error occured: {e}')

# 不发一张图证明你有娃（第一页）
page = 1

first_url = 'https://edith.xiaohongshu.com/api/sns/web/v2/comment/page?note_id=67cd493f0000000029034b9b&cursor=&top_comment_id=&image_formats=jpg,webp,avif&xsec_token=AB0kMXrfUC2RCEbz-ZzCpOzkWZZxro9CQlKplnvqro1Os%3D'
headers = {
    'Cookie': 'abRequestId=b4db5dae-661f-546b-a666-1a8ce2ae85c7; webBuild=4.61.1; xsecappid=xhs-pc-web; a1=195e9bfb447bmxxb82ms95tls6eb1cwq6na4k50hr50000426758; webId=30b0e6614ca83d675273e1158f816fca; acw_tc=0a4ab31917433834117097425e5bd211ad27ab80f6726dc597f6a3dd9a6708; gid=yj2djDiSDSK8yj2djDiD4I284WD6CCDYJ6Mj0xF291MKd428DySEAT8884JKW2Y8YSqqfJ4f; web_session=040069b57cee94f72269daa3dc354b8af528f0; websectiga=a9bdcaed0af874f3a1431e94fbea410e8f738542fbb02df1e8e30c29ef3d91ac; sec_poison_id=17408b0d-2557-4e12-98c4-905d7bcef109; loadts=1743383870484',
    # 'Cookie': 'abRequestId=3d2c77a2-f58d-5a9d-8286-e0fc73ad8eca; a1=195cfd97454ce4ota9flqtnna04b8li9x48kguddu50000217505; webId=e46e49f89708cfc0b3a5f6fd327dd4f2; gid=yj2SifjYYdCKyj2SifjW447f24Sd4l90ji1AYf79II084T28DY13jk888JyW28284J0jWiYD; webBuild=4.61.1; xsecappid=xhs-pc-web; web_session=040069b57cee94f7226903dfd1354b8cce9d9e; unread={%22ub%22:%2267dfad39000000001d0152e8%22%2C%22ue%22:%2267dc150c000000001d021d47%22%2C%22uc%22:21}; loadts=1743034477107; acw_tc=0a4ad72c17430346529807268e3bcc293fd0d3dbd9f319254ac3662ab9c2a2; websectiga=3633fe24d49c7dd0eb923edc8205740f10fdb18b25d424d2a2322c6196d2a4ad; sec_poison_id=f43db791-f521-4e14-8f8c-801d23807fe6',
    # 'Cookie': 'abRequestId=3d2c77a2-f58d-5a9d-8286-e0fc73ad8eca; a1=195cfd97454ce4ota9flqtnna04b8li9x48kguddu50000217505; webId=e46e49f89708cfc0b3a5f6fd327dd4f2; gid=yj2SifjYYdCKyj2SifjW447f24Sd4l90ji1AYf79II084T28DY13jk888JyW28284J0jWiYD; webBuild=4.61.1; xsecappid=xhs-pc-web; web_session=040069b57cee94f7226903dfd1354b8cce9d9e; unread={%22ub%22:%2267dfad39000000001d0152e8%22%2C%22ue%22:%2267dc150c000000001d021d47%22%2C%22uc%22:21}; loadts=1743034477107; acw_tc=0a4ad72c17430346529807268e3bcc293fd0d3dbd9f319254ac3662ab9c2a2; websectiga=9730ffafd96f2d09dc024760e253af6ab1feb0002827740b95a255ddf6847fc8; sec_poison_id=723e4925-ce7c-4ea2-a551-824a83c58257',
    # 'Host': 'xiaohongshu.com',
    'Referer': 'https://www.xiaohongshu.com/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
}
has_more = True
page_count = 0

# 如何确定爬取完成：has more如果为true，就可以一直爬
while has_more:
    page_count += 1
    print(f'第{page_count}页')
    if page_count == 1:
        url = first_url
    else:
        url = f'https://edith.xiaohongshu.com/api/sns/web/v2/comment/page?note_id=67cd493f0000000029034b9b&cursor={next_cursor}&top_comment_id=&image_formats=jpg,webp,avif&xsec_token=AB0kMXrfUC2RCEbz-ZzCpOzkWZZxro9CQlKplnvqro1Os%3D'

    resp = requests.get(url, headers=headers)
    resp.encoding = 'utf-8'  # 解决乱码问题
    pageSource = resp.text

    # print(pageSource)
    json_page = json.loads(pageSource)
    has_more = json_page['data']['has_more']
    next_cursor = json_page['data']['cursor']
    data = json_page['data']['comments']
    # print(len(data), data[0].keys())    # 每页总共有10条
    good_comments = {}
    randomImageId = 0
    for each_comment in data:
        id = each_comment['id']
        content = each_comment['content']
        like_count = each_comment['like_count']
        if 'pictures' in each_comment:
            pictures = each_comment['pictures']
            if len(pictures) <= 0:
                continue

            comment_picture_url = pictures[0]['url_default']
            comment_picture_with = pictures[0]['width']
            comment_picture_height = pictures[0]['height']
            if len(content.strip()) < 1:
                download_image(comment_picture_url, os.path.join(f'images_/{randomImageId}_{like_count}.jpg'))
                randomImageId += 1
            else:
                download_image(comment_picture_url, os.path.join(f'images_/{content}_{like_count}.jpg'))

        sub_comment_count = each_comment['sub_comment_count']
        sub_comments = each_comment['sub_comments']
        if len(sub_comments) > 0:
            # 只取like数最多的第一个
            sub_comment = sub_comments[0]
            comment_like_count = sub_comment['like_count']
            comment_content = sub_comment['content']
            # print(f'comment content: {comment_content}')
            # good_comments[id] = comment_content
            good_comments[randomImageId] = comment_content
        # print(f'sub comment count: {len(sub_comments)}')

for k,v in good_comments.items():
    print(k, v)
    # at_users, create_time, ip_location, status, pictures, note_id, user_info
    # ('liked'
    #  '',
    #  'sub_comment_cursor',
    #  'sub_comment_has_more',
    #  '',
    #  '',
    #  '',
    #  '',
    #  '',
    #  'show_tags',
    #  '',
    #  '',
    #  '')


