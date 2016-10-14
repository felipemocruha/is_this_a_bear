#coding: utf-8

import os
import cv2
from pymongo import MongoClient

if __name__ == '__main__':
    
    bear_img_paths = os.listdir('../images/bear/')
    not_bear_img_paths = os.listdir('../images/not_bear/')

    images_bear = []
    images_not_bear = []
    
    print 'Carregando imagens...'

    #bear loading
    for i in bear_img_paths:
        im = cv2.imread('../images/bear/'+i)
        if im != None:
            images_bear.append(im)
    
    for i in xrange(len(images_bear)):
        try:
            cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
        except Exception, e:
            print e

    #not_bear loading
    for i in bear_img_paths:
        im = cv2.imread('../images/not_bear/'+i)
        if im != None:
            images_not_bear.append(im)
    
    for i in xrange(len(images_not_bear)):
        try:
            cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
        except Exception, e:
            print e
    
    #SIFT
    desc_sift_bear = []
    desc_sift_not_bear = []
    
    sift = cv2.xfeatures2d.SIFT_create()
    print 'Criando os descritores SIFT...'
    for i in images_bear:
        (kps, desc) = sift.detectAndCompute(i, None)
        desc_sift_bear.append(desc)

    for i in images_not_bear:
        (kps, desc) = sift.detectAndCompute(i, None)
        desc_sift_not_bear.append(desc)

    #db connection
    client = MongoClient()
    db = client.is_this_a_bear
    img_descs = db.img_descs

    for i in desc_sift_bear:
        img = {"descriptor":i.tolist(),
               "is_this_a_bear":"yes"}
        img_descs.insert_one(img)

    for i in desc_sift_not_bear:
        img = {"descriptor":i.tolist(),
               "is_this_a_bear":"no"}
        img_descs.insert_one(img)
        
    print "db criado com sucesso!"
    
