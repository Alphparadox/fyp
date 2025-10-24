#!/usr/bin/env python3
# kiva_rule_transfer_and_select.py
import sys
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms

# ---------- Utility: align B to A using ORB + homography ----------
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

# ---------- Compute binary change mask and bounding boxes ----------
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

# ---------- Region analysis: classify change type ----------
from skimage.metrics import structural_similarity as sk_ssim
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
    varA = aGray.var()
    varB = bGray.var()
    s = 0.0
    try:
        s = sk_ssim(aGray, bGray)
    except:
        s = 0.0
    meanA = np.mean(a.reshape(-1,3), axis=0)
    meanB = np.mean(b.reshape(-1,3), axis=0)
    meanDiff = np.linalg.norm(meanB - meanA)
    change = {"bbox":bbox}
    # heuristics
    if varA < 200 and varB > 200 and meanDiff > 10:
        change["type"] = "insertion"
    elif varA > 200 and varB < 200 and meanDiff > 10:
        change["type"] = "removal"
    elif s > 0.6 and meanDiff > 15:
        change["type"] = "color_change"
        change["deltaMean"] = (meanB - meanA).tolist()
    else:
        shift = detectKeypointShift(a, b)
        if shift is not None:
            change["type"] = "movement"
            change["shift"] = shift
        else:
            # check multiplicity (number of connected components) for number change
            compA = countConnectedComponents(a)
            compB = countConnectedComponents(b)
            if compA != compB:
                change["type"] = "count_change"
                change["countA"] = compA
                change["countB"] = compB
            else:
                change["type"] = "complex"
    return change

def countConnectedComponents(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:
        if cv2.contourArea(c) > 50:
            count += 1
    return count

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

# ---------- Patch blending and helpers ----------
def createAlphaMask(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, a = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    a = cv2.GaussianBlur(a, (7,7), 0)
    alpha = (a.astype(np.float32) / 255.0)
    alpha = np.expand_dims(alpha, axis=2)
    return alpha

def blendPatch(img, patch, topleft, alpha):
    x,y = topleft
    h,w = patch.shape[:2]
    x2 = min(img.shape[1], x+w)
    y2 = min(img.shape[0], y+h)
    w = x2 - x
    h = y2 - y
    if w <= 0 or h <= 0:
        return
    patchC = patch[0:h,0:w].astype(np.float32)
    alphaC = alpha[0:h,0:w]
    region = img[y:y+h, x:x+w].astype(np.float32)
    blended = (patchC * alphaC + region * (1.0 - alphaC)).astype(np.uint8)
    img[y:y+h, x:x+w] = blended

def findTemplateInImage(img, template, minScore=0.45):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayTpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if grayTpl.shape[0] > grayImg.shape[0] or grayTpl.shape[1] > grayImg.shape[1]:
        return None
    res = cv2.matchTemplate(grayImg, grayTpl, cv2.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
    if maxVal < minScore:
        return None
    return maxLoc

def eraseRegion(img, x, y, w, h):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x,y), (x+w, y+h), 255, -1)
    res = cv2.inpaint(img.copy(), mask, 3, cv2.INPAINT_TELEA)
    return res

# ---------- Apply detected changes (A->B) to image C ----------
def applyChangesToC(imgC, changes, imgA, imgB):
    out = imgC.copy()
    hA, wA = imgA.shape[:2]
    hC, wC = imgC.shape[:2]
    scaleX = wC / float(wA)
    scaleY = hC / float(hA)
    for ch in changes:
        x,y,w,h = ch["bbox"]
        tgtX = int(round(x * scaleX))
        tgtY = int(round(y * scaleY))
        tgtW = int(round(w * scaleX))
        tgtH = int(round(h * scaleY))
        if tgtW <= 0 or tgtH <= 0:
            continue
        if ch["type"] == "insertion":
            srcPatch = imgB[y:y+h, x:x+w]
            if srcPatch.size == 0:
                continue
            srcRes = cv2.resize(srcPatch, (tgtW, tgtH), interpolation=cv2.INTER_AREA)
            blendPatch(out, srcRes, (tgtX, tgtY), createAlphaMask(srcRes))
        elif ch["type"] == "removal":
            out = eraseRegion(out, tgtX, tgtY, tgtW, tgtH)
        elif ch["type"] == "color_change":
            patch = out[tgtY:tgtY+tgtH, tgtX:tgtX+tgtW]
            if patch.size == 0:
                continue
            delta = np.array(ch.get("deltaMean", [0,0,0]))
            patch = np.clip(patch.astype(np.float32) + delta.reshape(1,1,3), 0, 255).astype(np.uint8)
            out[tgtY:tgtY+tgtH, tgtX:tgtX+tgtW] = patch
        elif ch["type"] == "movement":
            srcPatch = imgA[y:y+h, x:x+w]
            if srcPatch.size == 0:
                continue
            srcRes = cv2.resize(srcPatch, (tgtW, tgtH), interpolation=cv2.INTER_AREA)
            best = findTemplateInImage(out, srcRes)
            if best is not None:
                bx, by = best
                out = eraseRegion(out, bx, by, tgtW, tgtH)
                blendPatch(out, srcRes, (bx, by), createAlphaMask(srcRes))
            else:
                shift = ch.get("shift", (0.0,0.0))
                newX = tgtX + int(round(shift[0] * scaleX))
                newY = tgtY + int(round(shift[1] * scaleY))
                newX = max(0, min(newX, out.shape[1]-tgtW))
                newY = max(0, min(newY, out.shape[0]-tgtH))
                out = eraseRegion(out, tgtX, tgtY, tgtW, tgtH)
                blendPatch(out, srcRes, (newX, newY), createAlphaMask(srcRes))
        elif ch["type"] == "count_change":
            # simple heuristic: if count increased, attempt to copy the object from B's patch
            if ch.get("countB",0) > ch.get("countA",0):
                srcPatch = imgB[y:y+h, x:x+w]
                if srcPatch.size == 0:
                    continue
                srcRes = cv2.resize(srcPatch, (tgtW, tgtH), interpolation=cv2.INTER_AREA)
                # try to paste a copy at a nearby location
                pasteX = min(out.shape[1]-tgtW, max(0, tgtX + 10))
                pasteY = min(out.shape[0]-tgtH, max(0, tgtY + 10))
                blendPatch(out, srcRes, (pasteX, pasteY), createAlphaMask(srcRes))
            else:
                out = eraseRegion(out, tgtX, tgtY, tgtW, tgtH)
        else:
            # fallback: copy B patch into C
            srcPatch = imgB[y:y+h, x:x+w]
            if srcPatch.size == 0:
                continue
            srcRes = cv2.resize(srcPatch, (tgtW, tgtH), interpolation=cv2.INTER_AREA)
            blendPatch(out, srcRes, (tgtX, tgtY), createAlphaMask(srcRes))
    return out

# ---------- Similarity scoring: SSIM + VGG feature cosine ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        vggm = models.vgg16(pretrained=True).features.eval().to(device)
        for p in vggm.parameters():
            p.requires_grad = False
        vgg = vggm
    return vgg

def vggFeatureCosine(img1, img2):
    try:
        loadVGG()
        t1 = preprocess_vgg(img1).unsqueeze(0).to(device)
        t2 = preprocess_vgg(img2).unsqueeze(0).to(device)
        with torch.no_grad():
            f1 = vgg(t1).flatten(1)
            f2 = vgg(t2).flatten(1)
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            cos = (f1 * f2).sum().item()
        return float(cos)
    except Exception as e:
        return 0.0

def imageSSIM(img1, img2):
    try:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        g1s = cv2.resize(g1, (224,224))
        g2s = cv2.resize(g2, (224,224))
        score = sk_ssim(g1s, g2s)
        return float(score)
    except:
        return 0.0

# ---------- Full pipeline: detect, transfer, compare candidates ----------
def processAndSelect(pathA, pathB, pathC, candPaths, outTransformedPath=None):
    imgA = cv2.imread(pathA)
    imgB = cv2.imread(pathB)
    imgC = cv2.imread(pathC)
    if imgA is None or imgB is None or imgC is None:
        raise ValueError("Could not read one or more images.")
    alignedB, H = alignImages(imgA, imgB)
    mask, boxes = getChangeMask(imgA, alignedB)
    changes = []
    for b in boxes:
        ch = analyzeRegion(imgA, alignedB, b)
        changes.append(ch)
    transformedC = applyChangesToC(imgC, changes, imgA, alignedB)
    if outTransformedPath:
        Image.fromarray(cv2.cvtColor(transformedC, cv2.COLOR_BGR2RGB)).save(outTransformedPath)
    # compare to candidates
    scores = []
    for cp in candPaths:
        cand = cv2.imread(cp)
        if cand is None:
            scores.append((-9999.0, -9999.0))
            continue
        ssimScore = imageSSIM(transformedC, cand)
        vggScore = vggFeatureCosine(transformedC, cand)
        # combine (weight VGG higher if available)
        combined = 0.45 * ssimScore + 0.55 * ((vggScore + 1.0) / 2.0)  # map [-1,1] -> [0,1] then weight
        scores.append((combined, ssimScore))
    bestIdx = int(np.argmax([s[0] for s in scores]))
    return bestIdx, scores, changes, transformedC

# ---------- CLI ----------
def main():
    if len(sys.argv) < 8:
        print("Usage: python kiva_rule_transfer_and_select.py A.png B.png C.png cand1.png cand2.png cand3.png out_transformed.png")
        return
    pathA = sys.argv[1]
    pathB = sys.argv[2]
    pathC = sys.argv[3]
    cand1 = sys.argv[4]
    cand2 = sys.argv[5]
    cand3 = sys.argv[6]
    outp = sys.argv[7]
    bestIdx, scores, changes, transformed = processAndSelect(pathA, pathB, pathC, [cand1,cand2,cand3], outTransformedPath=outp)
    print("Selected candidate index (0-based):", bestIdx)
    print("Scores (combined, ssim):")
    for i,s in enumerate(scores):
        print(f"  Candidate {i}: combined={s[0]:.4f}, ssim={s[1]:.4f}")
    print("Detected changes:")
    for ch in changes:
        print(" ", ch)
    print("Transformed image saved to:", outp)

if __name__ == "__main__":
    main()