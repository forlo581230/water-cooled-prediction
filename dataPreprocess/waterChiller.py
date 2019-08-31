from CoolProp.CoolProp import PropsSI
import math


class WaterChiller:
    '''
    冷卻能力 219～877 kW
    '''
    def __init__(self):
        self.refrigerantMass = 185/122 #kg
        self.specificHeatOfWater = 4200 #J/kg-K
    '''
    Pc: condensor pressure (pa)
    Pe: evaporator pressure (pa)
    Tsuc: compressor suction temperature (K)
    Tdis: compressor discharge temperature (K)
    Tll: condensor outlet temperature (K)
    '''
    def getCOP(self, Pc=0, Pe=0, Tsuc=0, Tdis=0, Tll=0, P=0):
        '''
        h1, h2, h3, h4 : J/kg-K
        P : kW
        '''
        h1, h2, _, h4 = self.getEnthalpy(Pc, Pe, Tsuc, Tdis, Tll)
        # alpha = self.getCompressorEfficiency(Pc,Pe,Tsuc,Tdis,P)
        alpha = 0.95

        q = h1 - h4
        w = h2 - h1

        # refrigerant
        # 185 kg 變頻
        # 122 kg 定頻
        m = P*alpha / (w/1000)
        # m = P*alpha / (P)
        Q = m * (q/1000)

        COP = Q/P

        return COP, m, q/1000, P, alpha


    def getEnthalpy(self, Pc=0, Pe=0, Tsuc=0, Tdis=0, Tll=0):

        Tsuc = Tsuc + 273.15
        Tdis = Tdis + 273.15
        Tll = Tll + 273.15
        Pc = self.pressureTranslate(Pc)
        Pe = self.pressureTranslate(Pe)

        h2 = PropsSI('H','T',Tdis, 'P|gas', Pc,'R134a')
        h3 = PropsSI('H','T',Tll, 'P|liquid', Pc,'R134a')

        h4=h3

        h1 = PropsSI('H','T',Tsuc, 'P', Pe,'R134a')


        # plt.plot([h2, h1], [Pc, Pe])
        # plt.plot([h3, h4], [Pc, Pe])

        # plt.plot([h2, h3], [Pc, Pc])
        # plt.plot([h4, h1], [Pe, Pe])
        # plt.show()
        return h1, h2, h3, h4

    def getSuctionLineSuperheat(self, Tsuc, Pe):
        '''
        Tsh = Tsuc - Tev
        '''
        Pe = self.pressureTranslate(Pe)

        Tev = PropsSI('T','P',Pe, 'Q', 1,'R134a')- 273.15

        Tsh = Tsuc - Tev

        return Tsh, Tsuc, Tev

    def getLiquidLineSubcooling(self, Tcdo, Pc):
        '''
        Tsc = Tcd - Tcdo
        '''
        Pc = self.pressureTranslate(Pc)

        Tcd = PropsSI('T','P',Pc, 'Q', 0,'R134a')- 273.15

        Tsc = Tcd - Tcdo
        # print("expectd:",Tcd, "real:" ,Tcdo)

        return Tsc, Tcd, Tcdo
        
    
    def getCompressorEfficiency(self, Pc=0, Pe=0, Tsuc=0, Tdis=0, P=0):
        '''
        Pc, Pe : pascal
        Tsuc, Tdis : K(273.15)
        '''

        actual_h1 = PropsSI('H','T',Tsuc, 'P', Pe,'R134a')
        # print(Tsuc-273.15, Pe/1000,"***********")
        actual_h2 = PropsSI('H','T',Tdis, 'P', Pc,'R134a')
        
        actual_S1 = PropsSI('S','T',Tsuc, 'P|gas', Pe,'R134a')
        # actual_S1 = PropsSI('S','T',Tsuc, 'Q', 1,'R134a')

        expect_h2 = PropsSI('H','S',actual_S1, 'P|gas', Pc,'R134a')

        # validation
        actual_S2 = PropsSI('S','T',Tdis, 'P|gas', Pc,'R134a')
        expect_S2 = PropsSI('S','H',expect_h2, 'P|gas', Pc,'R134a')



        actual_diff_h1_h2 = actual_h2-actual_h1
        expect_diff_h1_h2 = expect_h2-actual_h1

        alpha = expect_diff_h1_h2/actual_diff_h1_h2
        # alpha = expect_diff_h1_h2/P/1000

        # print(actual_diff_h1_h2)
        if alpha >= 1:
            # print(actual_S1, actual_S2)
            # print(actual_S1, expect_S2,"(9999)")

            # TX = PropsSI('T','S',actual_S2, 'P|gas', Pe,'R134a')
            # print(Tsuc-273.15,TX-273.15)

            # print(actual_h1, actual_h2)
            # print(actual_h1, expect_h2,"(2)")
            # print(alpha,"=")
            alpha = 1

        return alpha
    
    


    def pressureTranslate(self, p):
        # kgf/cm^2 to pa
        # return p * (1/0.0102)*1000
        return p * 98068.059233108



# wc = WaterChiller()

# print(wc.getCOP_2(30.7*1000, 2.87, (17.5+15.5)/2, (31.1+31.7)/2, 17.1, 15.5))

# Mevap =wc.getEvaporatorWaterFlowRate2(PropsSI('P','T',5.4 + 273.15, 'Q', 1,'R134a'), 7.8, 34.3, 17.1, 15.5)
# Q = Mevap * 4.200 * (17.1 -  15.5)
# print(Q/30.7)

# Pc = PropsSI('P','T',35.6 + 273.15, 'Q', 0,'R134a')



# # 1 kpa = 0.0102 kgf/cm^2
# def pressureTranslate(p):
#     # kgf/cm^2 to pa
#     return p * (1/0.0102)*1000
# Te=11.1
# Tc=29.6
# Tevi=17.5
# Tevo=16.6
# Hll = PropsSI('H','T', Tc +273.15, 'Q', 0,'R134a')
# Hsuc = PropsSI('H','T', Te +273.15, 'P', pressureTranslate(3.35),'R134a')
# Mevap = (185/122*(Hsuc-Hll))/(4200 * (Tevi - Tevo))

# # print(Mevap)

# Pc=6.86
# Pe=2.88
# Tsuc=7.7
# Tdis=38.2
# Tll=30.1
# wc = WaterChiller()
# h1, h2, h4 = wc.getCOP_1(pressureTranslate(Pc), pressureTranslate(
#     Pe), Tsuc + 273.15, Tdis + 273.15, Tll + 273.15)

# print(h1)




import matplotlib.pyplot as plt
import numpy as np

def test():
    
    Pc=[pressureTranslate(8.43), pressureTranslate(7.14)]
    Pe=[pressureTranslate(2.61), pressureTranslate(3.3)]
    Tsuc= [5.4+273.15, 10.8+273.15]
    Tdis= [49.5+273.15, 37.1+273.15]
    Tll= [36.6+273.15, 31.4+273.15]
    # Tsuc = Tsuc +273.15
    # Tdis = Tdis +273.15
    # Tll = Tll +273.15

    wc = WaterChiller()

    hh=[]
    PP=[]
    for i in range(len(Pc)):
        h1, h2, h3, h4 = wc.getEnthalpy(Pc[i], Pe[i], Tsuc[i], Tdis[i], Tll[i])
        hh.append([0,h1, h2, h3, h4])
        PP.append([Pc[i], Pe[i]])

    hh=np.array(hh)
    PP=np.array(PP)
    # plt.plot([h2, h1], [Pc, Pe])
    # plt.plot([h3, h3], [Pc, Pe])

    # plt.plot([h2, h3], [Pc, Pc])
    # plt.plot([h3, h1], [Pe, Pe])
    for i in range(len(Pc)):
        print(hh[i,3])
        plt.plot([hh[i,2], hh[i,1]], [PP[i,0], PP[i,1]])
        plt.plot([hh[i,3], hh[i,3]], [PP[i,0], PP[i,1]])

        plt.plot([hh[i,2], hh[i,3]], [PP[i,0], PP[i,0]])
        plt.plot([hh[i,3], hh[i,1]], [PP[i,1], PP[i,1]])


    plt.show()

def test2():
    Pc=pressureTranslate(8.43)
    Pe=pressureTranslate(2.61)
    Tsuc= 5.4+273.15
    Tdis= 49.5+273.15
    Tll= 36.6+273.15
    # Tsuc = Tsuc +273.15
    # Tdis = Tdis +273.15
    # Tll = Tll +273.15

    wc = WaterChiller()

    hh=[]
    PP=[]

    h1, h2, h3, h4 = wc.getEnthalpy(Pc, Pe, Tsuc, Tdis, Tll)
    
    # 理想低壓狀態
    T1 = PropsSI('T','P',Pe, 'Q', 1,'R134a')
    # 理想高壓狀態
    T3 = PropsSI('T','P',Pc, 'Q', 0,'R134a')

    print(Tsuc-273.15, T1-273.15)
    print(Tll-273.15, T3-273.15)

# test2()