
import requests
from bs4 import BeautifulSoup
import urlparse
import os
import sys
import urllib
import text_file_formatter as tff

def get_image_links(image_number):
    base_url = 'https://s3.amazonaws.com/cizr-datasets-1/classifier-images-1/'
    url = '{}result{}'.format(base_url, image_number)
    return url

def download_image(link, filename):
    urllib.urlretrieve(link, filename)

if __name__ == '__main__':

    tup_lst = tff.file_reader('../links.txt')
    serve, InsidePoint, OutsidePoint = tff.label_split(tup_lst)
    sub_serve = tff.random_sample(serve,333)
    sub_InsidePoint = tff.random_sample(InsidePoint, 333)
    sub_OutsidePoint = tff.random_sample(OutsidePoint, 333)

    # for item in sub_serve:
    #     link = get_image_links(item[0])
    #     output_string = item[1]+item[0]
    #     download_image(link, output_string)

    # for item in sub_InsidePoint:
    #     link = get_image_links(item[0])
    #     output_string = item[1]+item[0]
    #     download_image(link, output_string)
    #
    for item in sub_OutsidePoint:
        link = get_image_links(item[0])
        output_string = item[1]+item[0]
        download_image(link, output_string)
