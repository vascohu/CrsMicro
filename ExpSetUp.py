from numpy import *
from math import *
import matplotlib.pyplot as plt

#PRICE_ARMS = arange(0.1,2.05,0.05);
PRICE_ARMS = arange(1,201,1);

from abc import ABCMeta, abstractmethod

# The class to define the mechanism interface
class Mechanism:
    __metaclass__ = ABCMeta;
    MECH_NAME = ["BP_UCB", "PD_BwK", "UCB_BwK", "BMCUB"];
    
    global PRICE_ARMS;
    
    def __init__(self, budget, initFocus):
        self.budget = budget;
        self.currentFocus = initFocus;
        self.costCurve = zeros(PRICE_ARMS.size);
        self.costCurve[initFocus:] = 1;
        self.sampleNum = zeros(PRICE_ARMS.size);
        self.counter = 0;
        
    @abstractmethod
    def postPrice(self):
        while False:
            yield None;
    
    @abstractmethod
    def receiveResult(self, accp_or_rej):
        while False:
            yield None;
            
    def getCostCurve(self):
        return self.costCurve;
    
    
class BP_UCB(Mechanism):
    MECH_NAME = "BP_UCB";
    
    def __init__(self, workerNum, budget, initFocus):
        Mechanism.__init__(self, budget, initFocus);
        self.BUDGET = budget;
        self.WRNUM = workerNum;
    
    def postPrice(self):
        if self.budget >= PRICE_ARMS[0]:
            self.counter+=1;
            tilF = zeros(self.costCurve.size);
            tilV = zeros(self.costCurve.size);
            for i in range(tilF.size):
                # compute $\tilde{F}^{t}_{i}$
                if self.sampleNum[i] == 0 :
                    tilF[i] = 1;
                else:
                    tilF[i] = self.costCurve[i] + sqrt(2*log(self.counter)/self.sampleNum[i]);

                # compute $\tilde{V}^{t}_{i}$
                tilV[i] = min(tilF[i], self.BUDGET/self.WRNUM/PRICE_ARMS[i]);
            # compute the optimal arm    
            maxArm = argmax(tilV);
            while (PRICE_ARMS[maxArm] > self.budget) & (maxArm >= 0) : maxArm -= 1;
            # return the price
            self.currentFocus = maxArm;
            self.sampleNum[maxArm] += 1;
            return PRICE_ARMS[maxArm];
        else:
            return 0;
        
    def receiveResult(self, accp_or_rej):
        self.costCurve[self.currentFocus] += (accp_or_rej - self.costCurve[self.currentFocus])/self.sampleNum[self.currentFocus];
        self.budget -= accp_or_rej * PRICE_ARMS[self.currentFocus];
        

class PD_BwK(Mechanism):
    MECH_NAME = "PD_BwK";
    K = PRICE_ARMS.size;
    v = ones(K);
    u = zeros(K);
    L = zeros(K);
    vK = 1;
    
    def __init__(self, workerNum, budget, initFocus):
        Mechanism.__init__(self, budget, initFocus);
        self.EP = sqrt(log(2)/min(workerNum, budget));
        self.Crad = log(workerNum*self.K*2);
    
    def postPrice(self):
        if self.budget >= PRICE_ARMS[0]:
            self.counter += 1;
            if self.counter <= PRICE_ARMS.size:
                maxArm = self.counter - 1;
            else:
                for i in range(self.K):
                    rad = sqrt(self.Crad*self.costCurve[i]/self.sampleNum[i]) + self.Crad/self.sampleNum[i];
                    self.u[i] = min(self.costCurve[i]+rad , 1.0);
                    self.L[i] = max((self.costCurve[i]-rad)*PRICE_ARMS[i] , 0.0);
                y = self.v/sum(self.v);
                F = divide(multiply(y , self.L) + self.vK, self.u);
                maxArm = argmin(F);
                
            # check the arm
            while (PRICE_ARMS[maxArm] > self.budget) & (maxArm >= 0) : maxArm -= 1;
            # update v
            self.v[maxArm] = self.v[maxArm]*power(1+self.EP,self.L[maxArm]);
            self.vK *= (1+self.EP);
            # return the price
            self.currentFocus = maxArm;
            self.sampleNum[maxArm] += 1;
            return PRICE_ARMS[maxArm];
        else:
            return 0;
        
    def receiveResult(self, accp_or_rej):
        self.costCurve[self.currentFocus] += (accp_or_rej - self.costCurve[self.currentFocus])/self.sampleNum[self.currentFocus];
        self.budget -= accp_or_rej * PRICE_ARMS[self.currentFocus];
        

from scipy.optimize import linprog

class UCB_BwK(Mechanism):
    MECH_NAME = "UCB_BwK";
    
    def __init__(self, workerNum, budget, initFocus):
        Mechanism.__init__(self, budget, initFocus);
        self.BUDGET = budget;
        self.WRNUM = workerNum;
        Delta = 0.01;
        Gamma = log(2*PRICE_ARMS.size*workerNum/Delta);
        self.EP = 0.1;#sqrt(Gamma*PRICE_ARMS.size/budget) + log(workerNum)*Gamma*PRICE_ARMS.size/budget;
        bounds = [];
        for i in range(PRICE_ARMS.size):
            bounds.append((0,1));
        self.prBounds = tuple(bounds);
    
    def postPrice(self):
        if self.budget >= PRICE_ARMS[0]:
            self.counter+=1;
            U = zeros(self.costCurve.size);
            L = zeros(self.costCurve.size);
            for i in range(PRICE_ARMS.size):
                # compute $upper confidence bound$
                if self.sampleNum[i] == 0 :
                    U[i] = 1;
                    L[i] = 0;
                else:
                    U[i] = min(self.costCurve[i] + sqrt(2*log(self.counter)/self.sampleNum[i]), 1.0);
                    L[i] = max(self.costCurve[i] - sqrt(2*log(self.counter)/self.sampleNum[i]), 0.0)*PRICE_ARMS[i];

            # compute the optimal distribution    
            c = U*(-1);
            A_ub = L;
            # print(U)
            # print(L)
            b_ub = [(1-self.EP)*self.BUDGET/self.WRNUM];
            A_eq = ones((1,PRICE_ARMS.size));
            b_eq = 1;
            pr = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=self.prBounds, options={"disp": False});
            #print(pr.x)
            maxArm = random.choice(PRICE_ARMS.size, 1, p=list(pr.x))[0];
            while (PRICE_ARMS[maxArm] > self.budget) & (maxArm >= 0) : maxArm -= 1;
            # return the price
            self.currentFocus = maxArm;
            self.sampleNum[maxArm] += 1;
            return PRICE_ARMS[maxArm];
        else:
            return 0;

        
    def receiveResult(self, accp_or_rej):
        self.costCurve[self.currentFocus] += (accp_or_rej - self.costCurve[self.currentFocus])/self.sampleNum[self.currentFocus];
        self.budget -= accp_or_rej * PRICE_ARMS[self.currentFocus];
        
from scipy.optimize import brentq
from sys import float_info        
class BMCUB(Mechanism):
    MECH_NAME = "BMCUB";
    
    def __init__(self, workerNum, budget, initFocus):
        Mechanism.__init__(self, budget, initFocus);
        self.BUDGET = budget;
        self.WRNUM = workerNum;
        self.Const = self.BUDGET/self.WRNUM;
        self.SwPNum = zeros(PRICE_ARMS.size);
        
    def __judgeFirstSwP(self, arm):
        '''Judge whether the point is the type-I switch point'''
        if self.costCurve[arm] < self.Const/PRICE_ARMS[arm]:
            if arm == (PRICE_ARMS.size - 1):
                return True;
            elif self.costCurve[arm] >= self.Const/PRICE_ARMS[arm+1]:
                return True;
            else:
                return False;
        else:
            return False;
    
    def __judgeSecondSwP(self, arm):
        '''Judge whether the point is the type-II switch point'''
        if self.costCurve[arm] >= self.Const/PRICE_ARMS[arm]:
            if arm == 0:
                self.SwPNum[arm] += 1; # for record
                return True;
            elif self.costCurve[arm-1] < self.Const/PRICE_ARMS[arm]:
                self.SwPNum[arm] += 1; # for record
                return True;
            else:
                return False;
        else:
            return False;
    
    def __calc_b(self, p, s, n):
        '''Calculate the upper confidence bound'''
        if (s == 0) | (p == 1):
            return 1.0;     # The upper bound of the output is 1.0
        else:
            f = (log(n) + 3*log(log(n)))/s;
            I = lambda theta1, theta2: theta1*log(theta1/theta2) + (1-theta1)*log((1-theta1)/(1-theta2));
            func = lambda q: I(p,q)-f;
            if p < float_info.epsilon:
                p = float_info.epsilon;
            if func(1-float_info.epsilon) < 0:
                return 1.0;      # This line is because of the requirements in the nonlinear root solver.
            else:
                b = brentq(func, a = p, b = 1 - float_info.epsilon)
                return b;
        
    def __decisionSecondSwP(self, arm):
        '''make decision for the type-II switch point'''
        if ((self.SwPNum[arm]-1)%2 == 0) | (arm == 0) :
            return arm;         # In 50% cases, we take the corresponding arm.
        else:
            if self.__calc_b(self.costCurve[arm-1], self.sampleNum[arm-1], self.counter) < self.Const/PRICE_ARMS[arm]:
                return arm;
            else:
                return arm-1;
                
                
    def postPrice(self):
        if self.budget >= PRICE_ARMS[0]:
            self.counter+=1;
            maxArm = -1;
            
            '''decide the search direction according to the comparison between the cost curve and the budget curve at the current point''' 
            if self.costCurve[self.currentFocus] < self.BUDGET/self.WRNUM/PRICE_ARMS[self.currentFocus]:
                for k in range(self.currentFocus, PRICE_ARMS.size):
                    if self.__judgeFirstSwP(k):     # Check whether the current point is the type-I switch point
                        maxArm = k;
                        break;
                    elif self.__judgeSecondSwP(k):  # Check whether the current point is the type-II switch point
                        maxArm = self.__decisionSecondSwP(k);
                        break;
            else:
                for k in range(self.currentFocus,-1, -1):
                    if self.__judgeFirstSwP(k):     # Check whether the current point is the type-I switch point
                        maxArm = k;
                        break;
                    elif self.__judgeSecondSwP(k):  # Check whether the current point is the type-II switch point
                        maxArm = self.__decisionSecondSwP(k);
                        break;
            '''check the budget'''
            while (PRICE_ARMS[maxArm] > self.budget) & (maxArm >= 0) : maxArm -= 1;
            '''return the price'''       
            self.currentFocus = maxArm;
            self.sampleNum[maxArm] += 1;                                 
            return PRICE_ARMS[maxArm];
        else:
            return 0;
        
    def receiveResult(self, accp_or_rej):
        self.costCurve[self.currentFocus] += (accp_or_rej - self.costCurve[self.currentFocus])/self.sampleNum[self.currentFocus];
        self.budget -= accp_or_rej * PRICE_ARMS[self.currentFocus];    

# The abstract class for defining the model of workers
class Worker:
    __metaclass__ = ABCMeta;
    
    WORKER_NAME = ["Simulation_Worker", "Discrete_Choice_Worker"];
    
    @abstractmethod
    def getProbability(self, price):
        while False:
            yield None;
    
    @abstractmethod
    def accept_or_reject(self, price):
        while False:
            yield None;
            
            
class Simulation_Worker(Worker):
    global PRICE_ARMS;
    
    def __init__(self, n):
        self.cost = random.uniform(PRICE_ARMS[0],PRICE_ARMS[-1]);
    
    def getProbability(self, price):
        Pr = (price - PRICE_ARMS[0])/(PRICE_ARMS[-1]-PRICE_ARMS[0]);
        if Pr > 1:
            Pr = 1;
        elif Pr < 0:
            Pr = 0;
        return Pr;
        
    def accept_or_reject(self, price):
        if self.cost <= price:
            return(1);
        else:
            return(0);
 

class Discrete_Choice_Worker(Worker):
    # PRICE_ARMS = arange(1,201,1);
    # BUDGET/WORKNUM = 10~50;
    
    def __init__(self, n):
        self.s = 15;
        self.b = -0.39;
        self.M =2000;
        
    def getProbability(self, price):
        return exp(price/self.s - self.b)/(exp(price/self.s - self.b) + self.M);
    
        
    def accept_or_reject(self, price):
        Pr = exp(price/self.s - self.b)/(exp(price/self.s - self.b) + self.M);
        temp = random.uniform(0,1);
        if temp <= Pr:
            return(1);
        else:
            return(0);
       
from sys import modules  
class CroSPlatform:

    def __init__(self, budget, workerNum, mechName, workerName):
        self.budget = budget;
        self.workerNum = workerNum;
        mechClass = getattr(modules[__name__], mechName);
        workerClass = getattr(modules[__name__], workerName);
        self.mechanism = mechClass(self.workerNum, self.budget, 1);
        # self.mechanism = BP_UCB(self.workerNum, self.budget, 1);
        # self.mechanism = BMCUB(self.workerNum, self.budget, 1);
        # self.mechanism = PD_BwK(self.workerNum, self.budget, 1);
        # self.mechanism = UCB_BwK(self.workerNum, self.budget, 1);
        self.workers = [];
        self.prices = [];
        self.accp_or_rej = [];
        
        for i in range(self.workerNum):
            #worker = Simulation_Worker(i);
            worker = workerClass(i);
            self.workers.append(worker);    
    
    def testMech(self):
        util = 0;
        i = 0;
        for worker in self.workers:
            i=i+1;
            #print(i);
            p = self.mechanism.postPrice();
            self.prices.append(p);
            a_or_r = worker.accept_or_reject(p);
            self.accp_or_rej.append(a_or_r)
            util += a_or_r;
            self.mechanism.receiveResult(a_or_r);
        return(util);


def OptimalFixPrice(PRICES, CDF, BUDGET, WRNUM):
    'compute the utility corresponding to the optimal fixed price'
    util = [];
    for i in range(len(PRICES)):
        util.append(min(CDF[i], BUDGET/WRNUM/PRICES[i]));
    print(util);
    return max(util);
    
    

def OptimalStrategy(PRICES, CDF, BUDGET, WRNUM):
    c = CDF*(-1);
    A_ub = multiply(CDF, PRICES);
    b_ub = [BUDGET/WRNUM];
    A_eq = ones((1,PRICES.size));
    b_eq = 1;
    bounds = [];
    while len(bounds) < PRICES.size: bounds.append((0,1));
    prBounds = tuple(bounds);
    pr = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=prBounds, options={"disp": False});
    return (dot(CDF, pr.x), pr.x)
    
def Test(pid, WRNNUM, util_q, mechName, workerName, ratio):
    util_l = []
    for i in range(100):
        Test1 = CroSPlatform(ratio*WRNNUM, WRNNUM, mechName, workerName);
        util = Test1.testMech();
        util_l.append(util);
    util_q.put((WRNNUM,util_l));

import multiprocessing as mtp
import pickle

workerNum = arange(1000,21000,1000);
util = zeros(workerNum.size);

def mechTest(mechIndex, workerIndex, ratio):
    mechName = Mechanism.MECH_NAME[mechIndex];
    workerName = Worker.WORKER_NAME[workerIndex];    
    print("Test on "+mechName+" Starts...");
    print("Worker is "+workerName+"...")
    util_q = mtp.Queue();
    procs = []

    for i in range(workerNum.size):
        p = mtp.Process(target=Test, args=(i, workerNum[i], util_q, mechName, workerName, ratio));
        procs.append(p);
        p.start();
    
    resultlist = [];
    for i in range(workerNum.size):
        resultlist.append(util_q.get())
        
    for p in procs:
        p.join();
    
    filename = mechName+"_result";
    with open(filename, 'wb') as fp:
        pickle.dump(resultlist, fp)
        
    print("Test on " + mechName + " is finished!")


def optTest(workerIndex, ratio):
    workerName = Worker.WORKER_NAME[workerIndex];
    CDF = zeros(PRICE_ARMS.size);
    for i in range(PRICE_ARMS.size):
        CDF[i] = workerName.getProbability(PRICE_ARMS[i]);
    WRNUM = 20000;
    BUDGET = WRNUM*ratio;
    Res1, Res2 = OptimalStrategy(PRICE_ARMS, CDF, BUDGET, WRNUM)
    Res3 = OptimalFixPrice(PRICE_ARMS, CDF, BUDGET, WRNUM)
    print(Res1)
    print(Res2)
    print(Res3)

import sys, getopt

def main(argv):
    mechtype = -1;
    workertype = -1;
    ratio = -1;
    try:
        opts, args = getopt.getopt(argv,"lpm:w:c:",["mechanism=","worker="])
    except getopt.GetoptError:
        print('test.py -m <mechanism index> -w <worker index> -c <budget/worker>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-l':
            print("Mechanisms: ", Mechanism.MECH_NAME)
            print("Workers: ", Worker.WORKER_NAME)
            sys.exit()
        elif opt in ("-m", "--mechanism index"):
            mechtype = int(arg)
        elif opt in ("-w", "--worker index"):
            workertype = int(arg)
        elif opt in ("-c", "--budget worker ratio"):
            ratio = float(arg)
        elif opt in ("-p", "--optimal cases"):
            mechtype = len(Mechanism.MECH_NAME);
    if min([mechtype,workertype,ratio])<0:
        print('test.py -m <mechanism index> -w <worker index> -c <budget/worker>')
        sys.exit(2);
    
    if mechtype < len(Mechanism.MECH_NAME):
        mechTest(mechtype, workertype, ratio);
    else:
        optTest(workertype, ratio);


if __name__ == "__main__":
    main(sys.argv[1:])
