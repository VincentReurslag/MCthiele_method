# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 21:25:15 2021

@author: Vincent Reurslag
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *


def MCthiele(x_feed,x_bot,x_top,q,R):
    print(
    """This program uses the McThiele method to determine the amount of
    theoretical stages needed in a distillation column.
          
    Made by: Vincent Reurslag""")
    
    data = data_manager()

    if q == 1:
        q = 0.999
    
    
    
    
    """###################################################################"""
    
    
    #plotting equilibrium line
    plt.figure()
    plt.plot(data[:,0],data[:,1],marker = "x")
    plt.plot([0,1],[0,1])
    
    #plotting bot,feed and top composition
    plt.plot([x_bot,x_bot],[x_bot,0],"k--")
    plt.plot([x_top,x_top],[x_top,0],"k--")
    plt.plot([x_feed,x_feed],[x_feed,0],"k--")
    plt.xlim([0,1])
    plt.ylim([0,1])
    
    #drawing q-line and rectifying/stripping line
    q_slope = q/(q-1)
    top_slope = R/(R+1)
    q_start = [x_feed,x_feed]
    top_start = [x_top,x_top]
    
    q_line = np.array(q_start,dtype = "float")
    top_line = np.array(top_start,dtype = "float")
    dx = 0.01
    q_new = np.array([q_start[0]-dx,q_start[1]-dx*q_slope],dtype="float")
    q_line = np.concatenate((q_line,q_new),axis=0)
    q_line = q_line.reshape([2,2])
    top_new = np.array([top_start[0]-dx,top_start[1]-dx*top_slope],dtype = "float")
    top_line = np.concatenate((top_line,top_new),axis = 0)
    top_line = top_line.reshape([2,2])
    
    while True:
        """computes the q-line and top-line and checks when they are almost equal by the
        specificed error"""
        dx += 0.01
        q_new = np.array([q_start[0]-dx,q_start[1]-dx*q_slope],dtype="float")
        q_new = q_new.reshape([1,2])
        q_line = np.concatenate((q_line,q_new),axis=0)
        
        top_new = np.array([top_start[0]-dx,top_start[1]-dx*top_slope],dtype = "float")
        top_new = top_new.reshape([1,2])
        top_line = np.concatenate((top_line,top_new),axis = 0)
        
        error = 0.02
        if dx > 1:
            break
    
    L1 = line(q_line[0,:],q_line[-1,:])
    L2 = line(top_line[0,:],top_line[-1,:])
    q_intersection = intersection(L1,L2)
    
    if q_intersection:
        print("Interesection: ",q_intersection)
    else:
        raise Exception("Can not find an intersection between q and top-line")
        
    plt.plot([q_start[0],q_intersection[0]],[q_start[1],q_intersection[1]])
    plt.plot([top_start[0],q_intersection[0]],[top_start[1],q_intersection[1]])
    plt.plot([x_bot,q_intersection[0]],[x_bot,q_intersection[1]])
    
    
    check = stages_up(data,q_intersection)
    if q_intersection[1] > check[1]:
        raise Exception("Can not draw stages, invalid input: Increase reflux")
    
    
    #drawing stages
    b1x = x_bot
    b2x = q_intersection[0]
    b1y = x_bot
    b2y = q_intersection[1]
    
    skip = True
    start_xy = [x_bot,x_bot]
    num_stages = 0
    plt.show
    plt.pause(0.1)
    while True:
        """One loop draws one entire stage by finding the start and end point
        of the line by using interpolation. The skip variable is needed to go from the
        bottom-line to the top-line"""
        end_xy = stages_up(data,start_xy)
        plt.plot([start_xy[0],end_xy[0]],[start_xy[1],end_xy[1]],"b")
        plt.pause(0.2)
        start_xy = end_xy.copy()
        
        if skip:
            x_tray = findYPoint(b1y,b2y,b1x,b2x,start_xy[1])
    
        if x_tray > q_intersection[0]:
            b1x = q_intersection[0]
            b2x = x_top
            b1y = q_intersection[1]
            b1y = x_top
            x_tray = x_top - findYPoint(b1y,b2y,b1x,b2x,start_xy[1]) + q_intersection[0]
            skip = False
        end_xy = [x_tray,start_xy[1]]
        plt.plot([start_xy[0],end_xy[0]],[start_xy[1],end_xy[1]],"b")
        start_xy = end_xy.copy()
        num_stages += 1
        plt.pause(0.2)
        if x_tray >= x_top:
            break
        
    plt.title(["Number of stages: ",num_stages])
    
def data_manager():
    """ Makes a numpy array of the incoming VLE data"""
    
    with open("VLEdata1.txt","r") as fo:
        data = fo.readlines()
    
    temp = []
    for line in data:
        line = line.split("\t")
        line[0] = line[0].replace(",",".")
        line[1] = line[1].replace(",",".")
        x = float(line[0])
        y = float(line[1])
        temp.append([x,y])
    
    temp.insert(0,[0,0])                        
    temp.append([1,1])    
    data = np.array(temp,dtype = "float")
    return data

def findYPoint(xa,xb,ya,yb,xc):
    m = (ya - yb) / (xa - xb)
    yc = (xc - xb) * m + yb
    return yc

def stages_up(data,start):
    """Returns the interesection point with the VLE diagram"""
    x_start = start[0]
    for i in range(len(data[:,0])):
        if data[i,0] < x_start:
            continue
        else:
            p1y = data[i-1,1]
            p2y = data[i,1]
            p1x = data[i-1,0]
            p2x = data[i,0]
            break
    
    y = findYPoint(p1x,p2x,p1y,p2y,x_start)
    return [x_start,y]


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False



root = Tk()

doc_label = Label(root,text =
                  """This program uses the McThiele method to determine the amount of
theoretical stages needed in a distillation column.
      
Made by: Vincent Reurslag""")

Feedx_label = Label(root, text = "Feed composition: ")
Feedx_input = Entry(root,width = 5)

Botx_label = Label(root, text = "Bot composition: ")
Botx_input = Entry(root,width = 5)

Topx_label = Label(root, text = "Top composition: ")
Topx_input = Entry(root,width = 5)

q_label = Label(root, text = "q-line: ")
q_input = Entry(root,width = 5)

R_label = Label(root, text = "Reflux ratio: (R)")
R_input = Entry(root,width = 5)

START = Button(root,text = "START", width = 20, height = 5,bg = "green",
               command =    lambda: MCthiele(
                           float(Feedx_input.get()), 
                           float(Botx_input.get()), 
                           float(Topx_input.get()),
                           float(q_input.get()),
                           float(R_input.get()))  )


#Grid statements
doc_label.grid(row = 0,column = 0,columnspan = 2)

Feedx_label.grid(row = 1,column = 0)
Feedx_input.grid(row = 1,column = 1)

Botx_label.grid(row = 2,column = 0)
Botx_input.grid(row = 2,column = 1)

Topx_label.grid(row = 3,column = 0)
Topx_input.grid(row = 3,column = 1)

q_label.grid(row = 4,column = 0)
q_input.grid(row = 4,column = 1)

R_label.grid(row = 5,column = 0)
R_input.grid(row = 5,column = 1)

START.grid(row = 6,column = 0,columnspan = 2)



root.mainloop()

#! /usr/bin/env python
"""
Created on Tue Dec 29 12:39:14 2020

@author: Vincent Reurslag
"""





"""###################Input section####################################"""




    

    
