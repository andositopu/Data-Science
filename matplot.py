#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import numpy as np


# In[14]:


x = np.linspace(1,5,11)


# In[15]:


x


# In[16]:


y = x **2


# In[17]:


y


# In[18]:


plt.plot (x,y,'r')
plt.xlabel('sumbu x')
plt.ylabel('sumbu y')
plt.title ('Contoh Data')
plt.show()


# In[19]:


#untuk membuat grafik garis putus-putus
plt.subplot(1,2,1)
plt.plot(x,y,'r--')


# In[33]:


#membuat grafik di dalam grafik
fig = plt.figure()

a = fig.add_axes([0.1,0.1,0.5,0.6])
b = fig.add_axes([0.2,0.3,0.3,0.2])


# In[35]:


fig = plt.figure()

a = fig.add_axes([0.1,0.1,0.5,0.6])
b = fig.add_axes([0.2,0.3,0.3,0.2])

a.plot(x,y,'g')
a.set_xlabel('sumbu x luar')
a.set_ylabel('sumbu y luar')
a.set_title('grafik luar')

b.plot(x,y,'r')
b.set_xlabel('sumbu x dalam')
b.set_ylabel('sumbu y dalam')
b.set_title('grafik dalam')


# In[41]:


#membuat 2 grafik saling terhubung
fig =plt.figure()
bebeb = fig.add_axes([0,0,1,1])
bebeb.plot(x, x ** 2, label ='Data x pangkat 2')
bebeb.plot(x, x ** 3, label ='Data x pangkat 3')

bebeb.legend(loc=0)


# In[ ]:


#untuk save grafik
fig.savefig('nama file.png', dpi= 100)

