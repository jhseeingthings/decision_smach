#! /usr/bin/env python
# -*- coding: utf-8 -*-


# remove duplicated points, rearrange points with direction and distance
def getBoundariesCenterLine(midPointsList1, midPointsList2, curX, curY, curYaw):
    preMassMidPointsList = []
    for i in range(len(midPointsList1)):
        preMassMidPointsList.append(midPointsList1[i])
    for i in range(len(midPointsList2)):
        preMassMidPointsList.append(midPointsList2[i])
    massMidPointsList = []
    # 去除重复点
    for i in range(len(preMassMidPointsList)):
        if preMassMidPointsList[i] not in massMidPointsList:
            massMidPointsList.append(preMassMidPointsList[i])
        else:
            pass
    centerLine = []
    curPointPositionX = curX
    curPointPositionY = curY
    curPointPositionYaw = curYaw
    pointList = []
    print(massMidPointsList)
    for i in range(len(massMidPointsList)):
        pointList.append(massMidPointsList[i])
    while (1):
        closestMidPointX, closestMidPointY = getClosestPoint(curPointPositionX, curPointPositionY, curPointPositionYaw,
                                                             pointList)
        if closestMidPointX == -1 and closestMidPointY == -1:
            break
        centerLine.append([closestMidPointX, closestMidPointY])
        pointList.remove([closestMidPointX, closestMidPointY])
        curPointPositionYaw = math.atan2(closestMidPointY - curPointPositionY, closestMidPointX - curPointPositionX)
        curPointPositionX = closestMidPointX
        curPointPositionY = closestMidPointY

    return centerLine




#########################
# extract center points from two boundaries.
def getBoundariesCenterPoints(boundary1, boundary2):
    boundaryPointsNumber1 = len(boundary1)
    boundaryPointsNumber2 = len(boundary2)
    midPointsList1 = []
    midPointsList2 = []
    for i in range(boundaryPointsNumber1):
        x1 = boundary1[i][0]
        y1 = boundary1[i][1]
        minDistance = 1000000
        for j in range(boundaryPointsNumber2):
            x2 = boundary2[j][0]
            y2 = boundary2[j][1]
            tempDistance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if (tempDistance < minDistance):
                minDistance = tempDistance
                nearestPointX = x2
                nearestPointY = y2
        midPointX = (x1 + nearestPointX) / 2
        midPointY = (y1 + nearestPointY) / 2
        midPointsList1.append([midPointX, midPointY])
    for i in range(boundaryPointsNumber2):
        x2 = boundary2[i][0]
        y2 = boundary2[i][1]
        minDistance = 1000000
        for j in range(boundaryPointsNumber1):
            x1 = boundary1[j][0]
            y1 = boundary1[j][1]
            tempDistance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if (tempDistance < minDistance):
                minDistance = tempDistance
                nearestPointX = x1
                nearestPointY = y1
        midPointX = (x2 + nearestPointX) / 2
        midPointY = (y2 + nearestPointY) / 2
        midPointsList2.append([midPointX, midPointY])
    return midPointsList1, midPointsList2


# find the nearest point ahead
def getClosestPoint(curPointPositionX, curPointPositionY, curPointPositionYaw, pointList):
    minDistance = 100000
    closestMidPointX = -1
    closestMidPointY = -1
    for i in range(len(pointList)):
        vehicle2MidPoint = np.array([pointList[i][0] - curPointPositionX, pointList[i][1] - curPointPositionY])
        vec_yaw = np.array([math.cos(curPointPositionYaw), math.sin(curPointPositionYaw)])
        cosAngel = np.dot(vehicle2MidPoint, vec_yaw) / np.linalg.norm(vehicle2MidPoint) / np.linalg.norm(vec_yaw)
        tempDistance = math.sqrt((pointList[i][0] - curPointPositionX) ** 2 + (pointList[i][1] - curPointPositionY) ** 2)
        if tempDistance < minDistance and cosAngel > 0.5:
            minDistance = tempDistance
            closestMidPointX = pointList[i][0]
            closestMidPointY = pointList[i][1]
    return closestMidPointX, closestMidPointY

