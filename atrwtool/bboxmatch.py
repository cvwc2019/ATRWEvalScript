import numpy as np
import json
import time

#TODO NMS

def match(gt,det):
    clock=time.time()
    det_bboxs=det['bboxs']
    det_bboxs=list(filter(lambda o:o['image_id'] in gt,det_bboxs))
    det['bboxs']=det_bboxs
    for bbox in det_bboxs:
        matchedgt,iou=iou_match(bbox,gt[bbox['image_id']])
        if matchedgt==None:
            eid,fid,query=-1,(-1,-1,-1),'sing'
        else:
            eid,fid,query=matchedgt['eid'],matchedgt['fid'],matchedgt['query']
        bbox['eid'],bbox['fid'],bbox['query']=eid,fid,query
        bbox['iou']=iou
    #det -> gt(eid) done
    #nms det <- gt
    #remap
    d={}
    for bbox in det_bboxs:
        imgid,eid=bbox['image_id'],bbox['eid']
        if imgid not in d:
            d[imgid]={}
        if eid not in d[imgid]:
            d[imgid][eid]=[]
        d[imgid][eid].append(bbox)
    det_bboxs=d
    #nms
    for imgid in det_bboxs:
        for eid in det_bboxs[imgid]:
            if len(det_bboxs[imgid][eid])>1:
                det_bboxs[imgid][eid]=det_bboxs[imgid][eid][np.argmax(map(lambda o:o['iou'],det_bboxs[imgid][eid]))]
            else:
                det_bboxs[imgid][eid]=det_bboxs[imgid][eid][0]
    temp=[]
    for imgid,bboxs in det_bboxs.items():
        for eid,bbox in bboxs.items():
            temp.append(bbox)
    det_bboxs=temp
    det['bboxs']=det_bboxs
    usedid=np.asarray(list(map(lambda o:o['bbox_id'],det_bboxs)))
    reid_res=det['reid_result']
    #clean up unused id
    print(len(reid_res))
    temp=[]
    for i,obj in enumerate(reid_res):
        k=obj['query_id']
        if k not in usedid:
            continue
            # reid_res.pop(i)
        else:
            ans=np.asarray(obj['ans_ids'])
            ans=ans[np.isin(ans,usedid)].tolist()
            # ans=list(filter(lambda x:x in usedid,ans))
            temp.append({'query_id':k,'ans_ids':ans})
    det['reid_result']=temp
    return det

def iou(bba,bbb):
    if min(bba[2],bbb[2])-max(bba[0],bbb[0])<0 or \
       min(bba[3],bbb[3])-max(bba[1],bbb[1])<0:
       return 0
    i=(min(bba[2],bbb[2])-max(bba[0],bbb[0]))*(min(bba[3],bbb[3])-max(bba[1],bbb[1]))
    u=(bba[2]-bba[0])*(bba[3]-bba[1])+(bbb[2]-bbb[0])*(bbb[3]-bbb[1])-i
    return i/u

def iou_match(bbox,gts):
    det=bbox
    detbbox=det['pos']#x,y,w,h
    detbbox=(detbbox[0],detbbox[1],detbbox[0]+detbbox[2],detbbox[1]+detbbox[3])
    gtbboxs=gts
    ious=list(map(lambda gtbbox:iou(detbbox,gtbbox['bbox']),gtbboxs))
    if max(ious)<0.5:
        return None,0
    matched=gts[np.argmax(ious)]
    return matched,max(ious)
