#!/usr/bin/env python3
# kiva_rule_transfer_and_select_gpu.py

import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

# ---------- Utility: align B to A ----------
def alignImages(imgRef, imgToAlign, maxFeatures=4000, goodMatchPct=0.15):
    grayRef = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
    grayTo = cv2.cvtColor(imgToAlign, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(maxFeatures)
    kps1, desc1 = orb.detectAndCompute(grayRef, None)
    kps2, desc2 = orb.detectAndCompute(grayTo, None)
    if desc1 is None or desc2 is None or len(kps1) < 4 or len(kps2) < 4:
        return imgToAlign.copy(), np.eye(3)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    numGood = int(len(matches) * goodMatchPct)
    matches = matches[:max(4, numGood)]
    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, m in enumerate(matches):
        pts1[i, :] = kps1[m.queryIdx].pt
        pts2[i, :] = kps2[m.trainIdx].pt
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    if H is None:
        return imgToAlign.copy(), np.eye(3)
    height, width = imgRef.shape[:2]
    aligned = cv2.warpPerspective(imgToAlign, H, (width, height), flags=cv2.INTER_LINEAR)
    return aligned, H

# ---------- Change Mask ----------
def getChangeMask(imgA, imgB, blurSize=5, minArea=200):
    diff = cv2.absdiff(imgA, imgB)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurSize, blurSize), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(th)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < minArea:
            continue
        x,y,w,h = cv2.boundingRect(c)
        boxes.append((x,y,w,h))
        cv2.rectangle(mask, (x,y), (x+w, y+h), 255, -1)
    return mask, boxes

# ---------- Region Analysis ----------
def countConnectedComponents(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) > 50)

def detectKeypointShift(patchA, patchB):
    grayA = cv2.cvtColor(patchA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(patchB, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(500)
    k1, d1 = orb.detectAndCompute(grayA, None)
    k2, d2 = orb.detectAndCompute(grayB, None)
    if d1 is None or d2 is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    if len(matches) < 8:
        return None
    ptsA = np.float32([k1[m.queryIdx].pt for m in matches])
    ptsB = np.float32([k2[m.trainIdx].pt for m in matches])
    shifts = ptsB - ptsA
    medianShift = np.median(shifts, axis=0)
    if np.linalg.norm(medianShift) < 2:
        return None
    return (float(medianShift[0]), float(medianShift[1]))

def analyzeRegion(imgA, imgB, bbox):
    x,y,w,h = bbox
    a = imgA[y:y+h, x:x+w]
    b = imgB[y:y+h, x:x+w]
    if a.size == 0 or b.size == 0:
        return {"type":"unknown", "bbox":bbox}
    try:
        aGray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        bGray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    except:
        return {"type":"unknown", "bbox":bbox}
    varA, varB = aGray.var(), bGray.var()
    ssim_val = sk_ssim(aGray, bGray) if aGray.shape == bGray.shape else 0.0
    meanA = np.mean(a.reshape(-1,3), axis=0)
    meanB = np.mean(b.reshape(-1,3), axis=0)
    meanDiff = np.linalg.norm(meanB - meanA)

    if varA < 200 and varB > 200 and meanDiff > 10:
        return {"type":"insertion","bbox":bbox}
    elif varA > 200 and varB < 200 and meanDiff > 10:
        return {"type":"removal","bbox":bbox}
    elif ssim_val > 0.6 and meanDiff > 15:
        return {"type":"color_change","bbox":bbox,"deltaMean":(meanB-meanA).tolist()}
    else:
        shift = detectKeypointShift(a, b)
        if shift is not None:
            return {"type":"movement","bbox":bbox,"shift":shift}
        else:
            compA, compB = countConnectedComponents(a), countConnectedComponents(b)
            if compA != compB:
                return {"type":"count_change","bbox":bbox,"countA":compA,"countB":compB}
    return {"type":"complex","bbox":bbox}

# ---------- VGG16 Similarity ----------
vgg = None
preprocess_vgg = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def loadVGG():
    global vgg
    if vgg is None:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.features.eval()
        vgg = model.features.to(device)
    return vgg

def vggFeatureCosine(img1, img2):
    try:
        vgg = loadVGG()
        t1 = preprocess_vgg(img1).unsqueeze(0).to(device)
        t2 = preprocess_vgg(img2).unsqueeze(0).to(device)
        with torch.no_grad():
            f1 = vgg(t1).flatten(1)
            f2 = vgg(t2).flatten(1)
            f1, f2 = F.normalize(f1, dim=1), F.normalize(f2, dim=1)
            cos = (f1 * f2).sum().item()
        return float(cos)
    except:
        return 0.0

# ---------- Entrypoint ----------
if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: python kiva_rule_transfer_and_select_gpu.py A.png B.png C.png cand1.png cand2.png cand3.png out.png")
        sys.exit(1)
    print("[INFO] Starting processing...")
    from kiva_rule_transfer_and_select import processAndSelect
    pathA, pathB, pathC, cand1, cand2, cand3, outp = sys.argv[1:8]
    bestIdx, scores, changes, transformed = processAndSelect(pathA, pathB, pathC, [cand1,cand2,cand3], outTransformedPath=outp)
    print(f"[RESULT] Selected candidate index: {bestIdx}")
