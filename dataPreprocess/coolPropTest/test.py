import CoolProp.CoolProp as CoolProp

from CoolProp.CoolProp import PropsSI

for k in ['formula','CAS','aliases','ASHRAE34','REFPROP_name','pure','INCHI','INCHI_Key','CHEMSPIDER_ID']:
    item = k + ' --> ' + CoolProp.get_fluid_param_string("R134a", k)
    print(item)

# H_L = PropsSI('H','P',101325,'Q',0,'Water'); print(H_L)
# J/kg-K
# T = 597.9 K and P = 5.0e6 Pa
# Q = 1(vapor) =0(liquid)
T1 = 7.8
T2=51.9
T3=34.3
Tc=35.6
Te=5.4

Pc = PropsSI('P','T',Tc + 273.15, 'Q', 0,'R134a')
h2 = PropsSI('H','T',T2 + 273.15, 'P', Pc,'R134a')
print(h2)

h3 = PropsSI('H','T',T3+ 273.15, 'Q', 0,'R134a')
print(h3)
h4=h3

Pe = PropsSI('P','T',Te + 273.15, 'Q', 1,'R134a')
h1 = PropsSI('H','T',T1 + 273.15, 'P', Pe,'R134a')
print(h1)

print((h1-h4)/(h2-h1))