#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random,math,copy
import numpy as np
import pandas as pd
def Expectation(Function,DomainX,DomainY,ProbDistribution):
	CumSum=0
	for x in DomainX:
		for y in DomainY:
			CumSum+=Function(x,y)*ProbDistribution.loc[x,y]
	return CumSum
def Covariance(FunctionX,FunctionY,DomainX,DomainY,ProbDistribution):
	ExpFX=Expectation(FunctionX,DomainX,DomainY,ProbDistribution)
	ExpFY=Expectation(FunctionY,DomainX,DomainY,ProbDistribution)
	CumSum=0
	for x in DomainX:
		for y in DomainY:
			CumSum+=(FunctionX(x,y)-ExpFX)*(FunctionY(x,y)-ExpFY)*ProbDistribution.loc[x,y]
	return CumSum
def Func_x(x,y):
	return float(x)
def Func_y(x,y):
	return float(y)
def ConditionalProbDist(ProbDistribution,ConditionedOn):
	ConditionalDistribution=copy.copy(ProbDistribution)
	if ConditionedOn=='x':
		for x in DomainX:
			MarginalX=0
			for y in DomainY:
				MarginalX+=ProbDistribution.loc[x,y]
			for y in DomainY:
				ConditionalDistribution[y][x]=ProbDistribution.loc[x,y]/MarginalX
	elif ConditionedOn=='y':
		for y in DomainY:
			MarginalY=0
			for x in DomainX:
				MarginalY+=ProbDistribution.loc[x,y]
			for x in DomainX:
				ConditionalDistribution[y][x]=ProbDistribution.loc[x,y]/MarginalY
	return ConditionalDistribution
def ConditionalExpectation(Function,DomainX,DomainY,ProbDistribution,ConditionedOn):
	ConditionalDistribution=ConditionalProbDist(ProbDistribution,ConditionedOn)
	if ConditionedOn=='x':
		Results=np.reshape([0]*len(ProbDistribution.index),(1,len(ProbDistribution.index)))
		Results=pd.DataFrame(Results, columns=ProbDistribution.index)
		for x in DomainX:
			CumSum=0
			for y in DomainY:
				CumSum+=Function(x,y)*ConditionalDistribution.loc[x,y]
			Results[x]=CumSum

	elif ConditionedOn=='y':
		Results=np.reshape([0]*len(ProbDistribution.columns),(1,len(ProbDistribution.columns)))
		Results=pd.DataFrame(Results,columns=ProbDistribution.columns)
		for y in DomainY:
			CumSum=0
			for x in DomainX:
				CumSum+=Function(x,y)*ConditionalDistribution.loc[x,y]
			Results[y]=CumSum
	return Results
def JointEntropy(DomainX,DomainY,ProbDistribution):
	CumSum=0
	for x in DomainX:
		for y in DomainY:
			CumSum+=-np.log(ProbDistribution.loc[x,y])*ProbDistribution.loc[x,y] if ProbDistribution.loc[x,y] != 0 else 0
	return CumSum
def MarginalEntropy(DomainX,DomainY,ProbDistribution,MarginalizedOn):
	CumSum=0
	if MarginalizedOn=='x':
		for x in DomainX:
			MarginalX=0
			for y in DomainY:
				MarginalX+=ProbDistribution.loc[x,y]
			CumSum+=-np.log(MarginalX)*MarginalX if MarginalX != 0 else 0
	if MarginalizedOn=='y':
		for y in DomainY:
			MarginalY=0
			for x in DomainX:
				MarginalY+=ProbDistribution.loc[x,y]
			CumSum+=-np.log(MarginalY)*MarginalY if MarginalY != 0 else 0
	return CumSum
def ConditionalEntropies(DomainX,DomainY,ProbDistribution,ConditionalProbDist):
	CumSum=0
	for x in DomainX:
		for y in DomainY:
			CumSum+=-np.log(ConditionalProbDist.loc[x,y])*ProbDistribution.loc[x,y] if ConditionalProbDist.loc[x,y] != 0 and ProbDistribution.loc[x,y] !=0 else 0
	return CumSum
def MutualInformation(DomainX,DomainY,ProbDistribution,MarginalizedOn,ConditionalProbDist):
	return MarginalEntropy(DomainX,DomainY,ProbDistribution,MarginalizedOn)-ConditionalEntropies(DomainX,DomainY,ProbDistribution,ConditionalProbDist)
row_names=['1', '2']
column_names=['-1', '0', '5']
ProbMatrix=np.reshape((0.3, 0.3, 0, 0.1, 0.2, 0.1), (2, 3))
ProbDistribution=pd.DataFrame(ProbMatrix, columns=column_names, index=row_names)
DomainX=list(ProbDistribution.index)
DomainY=list(ProbDistribution.columns)

print('E[X]={0}'.format(round(Expectation(Func_x,DomainX,DomainY,ProbDistribution),4)))
print('E[Y]={0}'.format(round(Expectation(Func_y,DomainX,DomainY,ProbDistribution),4)))
print('E[X|Y]=\n{0}'.format(round(ConditionalExpectation(Func_x,DomainX,DomainY,ProbDistribution,'y'),4)))
print('E[Y|X]=\n{0}'.format(round(ConditionalExpectation(Func_y,DomainX,DomainY,ProbDistribution,'x'),4)))
print('Cov[X,Y]={0}'.format(round(Covariance(Func_x,Func_y,DomainX,DomainY,ProbDistribution),4)))
print('H[X,Y]={0}'.format(round(JointEntropy(DomainX,DomainY,ProbDistribution),4)))
print('H[X]={0}'.format(round(MarginalEntropy(DomainX,DomainY,ProbDistribution,'x'),4)))
print('H[Y]={0}'.format(round(MarginalEntropy(DomainX,DomainY,ProbDistribution,'y'),4)))
print('H[Y|X]={0}'.format(round(ConditionalEntropies(DomainX,DomainY,ProbDistribution,ConditionalProbDist(ProbDistribution,'x')),4)))
print('H[X|Y]={0}'.format(round(ConditionalEntropies(DomainX,DomainY,ProbDistribution,ConditionalProbDist(ProbDistribution,'y')),4)))
print('I(X,Y)={0}'.format(round(MutualInformation(DomainX,DomainY,ProbDistribution,'x',ConditionalProbDist(ProbDistribution,'y')),4)))
print('H[X,Y]={0}={1}+{2}=H[X]+H[Y|X]'.format(round(JointEntropy(DomainX,DomainY,ProbDistribution),4),
round(MarginalEntropy(DomainX,DomainY,ProbDistribution,'x'),4),
round(ConditionalEntropies(DomainX,DomainY,ProbDistribution,ConditionalProbDist(ProbDistribution,'x')),4)))
print('H[X,Y]={0}={1}+{2}=H[Y]+H[X|Y]'.format(round(JointEntropy(DomainX,DomainY,ProbDistribution),4),
round(MarginalEntropy(DomainX,DomainY,ProbDistribution,'y'),4),
round(ConditionalEntropies(DomainX,DomainY,ProbDistribution,ConditionalProbDist(ProbDistribution,'y')),4)))
print('H[X,Y]={0}={2}+{1}+{3}=H[X|Y]+H[X|Y]+I[X,Y]'.format(round(JointEntropy(DomainX,DomainY,ProbDistribution),4),
round(ConditionalEntropies(DomainX,DomainY,ProbDistribution,ConditionalProbDist(ProbDistribution,'x')),4),
round(ConditionalEntropies(DomainX,DomainY,ProbDistribution,ConditionalProbDist(ProbDistribution,'y')),4),
round(MutualInformation(DomainX,DomainY,ProbDistribution,'x',ConditionalProbDist(ProbDistribution,'y')),4)))
print('H[X]={0}={1}+{2}=H[X|Y]+I[X,Y]'.format(round(MarginalEntropy(DomainX,DomainY,ProbDistribution,'x'),4),
round(ConditionalEntropies(DomainX,DomainY,ProbDistribution,ConditionalProbDist(ProbDistribution,'y')),4),
round(MutualInformation(DomainX,DomainY,ProbDistribution,'x',ConditionalProbDist(ProbDistribution,'y')),4)))
print('H[Y]={0}={1}+{2}=H[Y|X]+I[X,Y]'.format(round(MarginalEntropy(DomainX,DomainY,ProbDistribution,'y'),4),
round(ConditionalEntropies(DomainX,DomainY,ProbDistribution,ConditionalProbDist(ProbDistribution,'x')),4),
round(MutualInformation(DomainX,DomainY,ProbDistribution,'x',ConditionalProbDist(ProbDistribution,'y')),4)))