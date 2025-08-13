# coding: utf-8
 

from __future__ import division
import numpy as np
import random
import math
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import re
from ollama import chat
import os
from scipy.io import savemat

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

def format_number(x, decimals=2):
    """格式化数字为字符串，控制小数位数"""
    if isinstance(x, float):
        # 浮点数：格式化为指定小数位
        return f"{x:.{decimals}f}"
    else:
        # 整数：直接转换为字符串
        return str(x)



class GA(object):
    def __init__(self, maxiter, sizepop, lenchrom, pc, pm, dim, lb, ub, Fobj):
        """
        maxiter：最大迭代次数
        sizepop：种群数量
        lenchrom：染色体长度
        pc：交叉概率
        pm：变异概率
        dim：变量的维度
        lb：最小取值
        ub：最大取值
        Fobj：价值函数
        """
        self.maxiter = maxiter
        self.sizepop = sizepop
        self.lenchrom = lenchrom
        self.pc = pc
        self.pm = pm
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.Fobj = Fobj
        self.len = 5
    def new_sub_pop(self, pop, fitness, num_new, dim):
        # 选择适应度较高的个体
        pop_new = []
        pop1 = np.array(copy.deepcopy(pop))



        combined = "\n".join(f'point:</solution> {",".join(map(format_number, lst))} </solution>, objective: {val:.2}' for lst, val in zip(pop1, fitness))


        # print(combined)


        content = '''You are an expert in math. The variable points will be represented in the following form: </solution>0.1,0.2,0.4,0.5</solution> with their objective values, where lower values are better. '''
        content = content + '\n' + combined + '\n' +  f'''Give me {num_new} new points with length {dim} that are diferent from all the points above.  Do not write code! Do not give any explanation!'''
        content = content  + '\n' +  f'''The length of new point must be {dim}! Each output new point must start with </solution> and end with </solution>!'''

        # print(content)
        response = chat(
        model='qwen2.5:14b',
        messages=[{'role': 'user', 'content': content }],
        )
        print(response['message']['content'])

        result = []
        for match in re.findall(r'</solution>(.*?)</solution>', response['message']['content']):
            # 将每个匹配项分割并转换为浮点数
            numbers = [float(x) for x in match.split(',')]
            print(len(numbers))
            result.append(numbers)

 

        for ii in range(len(result)):
            tmp = []
            if len(result[ii])==dim:
                for jj in range(dim):
                    
                    tmp.append(self.decimal_to_binary(result[ii][jj],self.lenchrom, self.lb[jj], self.ub[jj]))

                pop_new.append(tmp)

        return pop_new

    def decimal_to_binary(self, decimal_value, chromosome_length, min_val=-1.0, max_val=1.0):
        """
        将[-1,1]区间内的十进制值转换为二进制染色体
        
        参数:
        decimal_value -- 要转换的十进制值(单个值)
        chromosome_length -- 染色体长度(二进制位数)
        min_val -- 区间最小值, 默认为-1.0
        max_val -- 区间最大值, 默认为1.0
        
        返回:
        二进制字符串表示(染色体)
        """
        # 确保值在有效范围内
        clipped_value = np.clip(decimal_value, min_val, max_val)
        
        # 将值从[min_val, max_val]映射到[0, 2^chromosome_length - 1]
        normalized_value = (clipped_value - min_val) / (max_val - min_val)
        integer_value = int(normalized_value * (2**chromosome_length - 1))
        
        # 转换为二进制字符串并填充到指定长度
        binary_string = bin(integer_value)[2:].zfill(chromosome_length)  # 去掉'0b'前缀
        binary_list = [int(bit) for bit in binary_string]
        return binary_list

    # 初始化种群：返回一个三维数组，第一维是种子，第二维是变量维度，第三维是编码基因
    def Initialization(self):
        pop = []
        for i in range(self.sizepop):
            temp1 = []
            for j in range(self.dim):
                temp2 = []
                for k in range(self.lenchrom):
                    temp2.append(random.randint(0, 1))
                temp1.append(temp2)
            pop.append(temp1)
        return pop

    # 将二进制转化为十进制
    def b2d(self, pop_binary):
        pop_decimal = []
        for i in range(len(pop_binary)):
            temp1 = []
            for j in range(self.dim):
                temp2 = 0
                for k in range(self.lenchrom):
                    temp2 += pop_binary[i][j][k] * math.pow(2, k)
                temp2 = temp2 * (self.ub[j] - self.lb[j]) / (math.pow(2, self.lenchrom) - 1) + self.lb[j]
                temp1.append(temp2)
            pop_decimal.append(temp1)
        return pop_decimal

    # 轮盘赌模型选择适应值较高的种子
    def Roulette(self, fitness, pop):
        # 适应值按照大小排序
        sorted_index = np.argsort(fitness)
        sorted_fitness, sorted_pop = [], []
        for index in sorted_index:
            sorted_fitness.append(fitness[index])
            sorted_pop.append(pop[index])

        # 生成适应值累加序列
        fitness_sum = sum(sorted_fitness)
        accumulation = [None for col in range(len(sorted_fitness))]
        accumulation[0] = sorted_fitness[0] / fitness_sum
        for i in range(1, len(sorted_fitness)):
            accumulation[i] = accumulation[i - 1] + sorted_fitness[i] / fitness_sum

        # 轮盘赌
        roulette_index = []
        for j in range(len(sorted_fitness)):
            p = random.random()
            for k in range(len(accumulation)):
                if accumulation[k] >= p:
                    roulette_index.append(k)
                    break
        temp1, temp2 = [], []
        for index in roulette_index:
            temp1.append(sorted_fitness[index])
            temp2.append(sorted_pop[index])
        newpop = [[x, y] for x, y in zip(temp1, temp2)]
        newpop.sort()
        newpop_fitness = [newpop[i][0] for i in range(len(sorted_fitness))]
        newpop_pop = [newpop[i][1] for i in range(len(sorted_fitness))]
        return newpop_fitness, newpop_pop

    # 交叉繁殖：针对每一个种子，随机选取另一个种子与之交叉。
    # 随机取种子基因上的两个位置点，然后互换两点之间的部分
    def Crossover(self, pop):
        newpop = []
        for i in range(len(pop)):
            if random.random() < self.pc:
                # 选择另一个种子
                j = i
                while j == i:
                    j = random.randint(0, len(pop) - 1)
                cpoint1 = random.randint(1, self.lenchrom - 1)
                cpoint2 = cpoint1
                while cpoint2 == cpoint1:
                    cpoint2 = random.randint(1, self.lenchrom - 1)
                cpoint1, cpoint2 = min(cpoint1, cpoint2), max(cpoint1, cpoint2)
                newpop1, newpop2 = [], []
                for k in range(self.dim):
                    temp1, temp2 = [], []
                    temp1.extend(pop[i][k][0:cpoint1])
                    temp1.extend(pop[j][k][cpoint1:cpoint2])
                    temp1.extend(pop[i][k][cpoint2:])
                    temp2.extend(pop[j][k][0:cpoint1])
                    temp2.extend(pop[i][k][cpoint1:cpoint2])
                    temp2.extend(pop[j][k][cpoint2:])
                    newpop1.append(temp1)
                    newpop2.append(temp2)
                newpop.extend([newpop1, newpop2])
        return newpop



    # 变异：针对每一个种子的每一个维度，进行概率变异，变异基因为一位
    def Mutation(self, pop):
        newpop = copy.deepcopy(pop)
        for i in range(len(pop)):
            for j in range(self.dim):
                if random.random() < self.pm:
                    mpoint = random.randint(0, self.lenchrom - 1)
                    newpop[i][j][mpoint] = 1 - newpop[i][j][mpoint]
        return newpop

    # 绘制迭代-误差图
    def Ploterro(self, Convergence_curve):
        # mpl.rcParams['font.sans-serif'] = ['Courier New']
        mpl.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(10, 6))
        x = [i for i in range(len(Convergence_curve))]
        plt.plot(x, Convergence_curve, 'r-', linewidth=1.5, markersize=5)
        plt.xlabel(u'Iter', fontsize=18)
        plt.ylabel(u'Best score', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(0, )
        plt.grid(True)
        plt.savefig("Convergence_curve_LLM.png", dpi=300)
        res = {'x':x, 'y':Convergence_curve}
        savemat("Convergence_curve_LLM.mat", res)

    def Run(self):
        pop = self.Initialization()
        errolist = []
        for Current_iter in range(self.maxiter):
            print("Iter = " + str(Current_iter))
            pop1 = self.Crossover(pop)
            pop2 = self.Mutation(pop1)
            pop3 = self.b2d(pop2)
            fitness = []
            for j in range(len(pop2)):
                fitness.append(self.Fobj(pop3[j]))
            sorted_fitness, sorted_pop = self.Roulette(fitness, pop2)
            best_fitness = sorted_fitness[-1]
            best_pos = self.b2d([sorted_pop[-1]])[0]
            pop = sorted_pop[-1:-(self.sizepop + 1):-1]
            errolist.append(1 / best_fitness)
            if 1 / best_fitness < 0.0001:
                print("Best_score = " + str(round(1 / best_fitness, 4)))
                print("Best_pos = " + str([round(a, 4) for a in best_pos]))
                break

            new_pop = self.new_sub_pop(pop3[1:10], fitness[1:10], self.len, self.dim)
            if len(new_pop) > 0:
                pop[-len(new_pop):] = new_pop

        return best_fitness, best_pos, errolist


if __name__ == "__main__":
    # Set random seed
    set_seed(1)

    K = 2
    Nt = 6
    H_all = np.load('H_all.npy')
    # H_all = np.ones((Nt, K)) + 1j * np.ones((Nt, K))
    # a = (- np.ones((Nt * K * 2, ))).tolist()
    # b = np.ones((Nt * K * 2, )).tolist()
    # 价值函数，求函数最小值点 -> [1, -1, 0, 0]
    def Fobj(factor, Nt = Nt, K = K, H = H_all):
        w = (np.array(factor[:Nt*K]).reshape((Nt, K)) + 1j* np.array(factor[Nt*K:]).reshape((Nt, K)))
        # print(H_all)
        sigma = 1e-3
        Rk = 0
        for k in range(K):
            for jj in filter(lambda x: x != k, range(0, K)):
                Rk += np.log2(1 + np.abs(H_all[:,k].conj().reshape(1,-1) @ w[:,k].reshape(-1,1))**2 / (np.abs(H_all[:,k].conj().reshape(1,-1) @ w[:,jj].reshape(-1,1))**2) + sigma)

        return 1/Rk[0,0]
    
    starttime = time.time()
    a = GA(100, 50, 6, 0.8, 0.01, Nt * K * 2, (-np.ones((Nt * K * 2, ))).tolist(), np.ones((Nt * K * 2, )).tolist(), Fobj)
    Best_score, Best_pos, errolist = a.Run()
    endtime = time.time()
    print("Runtime = " + str(endtime - starttime))
    a.Ploterro(errolist)