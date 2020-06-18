#! /usr/bin/env python
# -*- coding: utf-8 -*-


class BoundaryPoints():
    def __init__(self, x):
        self.x = x


class LaneInfo:
    def __init__(self, road_msg):
        self.lanes = road_msg.lanes

bound_x = []
bound_y = []
lanes_nearby = LaneInfo()
for i in range(len(lanes_nearby.lanes)):
    for j in range(len(boundaries)):
        for k in range(len(boundaries[j].boundaryPoints)):
