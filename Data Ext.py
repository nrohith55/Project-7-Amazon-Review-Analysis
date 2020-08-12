#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bs4 import BeautifulSoup as bs
import requests


# In[ ]:


bt='https://www.amazon.in'
ul = 'https://www.amazon.in/MJSXJ02CM-1080P-Security-Camera-White/product-reviews/B07HJD1KH4/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'


# In[ ]:


cust_name = []
review_title = []
rate = []
review_content = []


# In[ ]:


tt = 0
while tt <= 420:
    page = requests.get(ul)
    while page.ok == False:
        page = requests.get(ul)
    
    soup = bs(page.content,'html.parser')
    soup.prettify()
    
    names = soup.find_all('span', class_='a-profile-name')
    names.pop(0)
    names.pop(0)
    
    for i in range(0,len(names)):
        cust_name.append(names[i].get_text())
        
    title = soup.find_all("a",{"data-hook":"review-title"})
    for i in range(0,len(title)):
        review_title.append(title[i].get_text())

    rating = soup.find_all('i',class_='review-rating')
    rating.pop(0)
    rating.pop(0)
    for i in range(0,len(rating)):
        rate.append(rating[i].get_text())

    review = soup.find_all("span",{"data-hook":"review-body"})
    for i in range(0,len(review)):
        review_content.append(review[i].get_text())
        
    for div in soup.findAll('li', attrs={'class':'a-last'}):
        A = div.find('a')['href']
    ul = bt + A
    
    tt = tt+1


# In[ ]:


len(cust_name)


# In[ ]:


len(review_title)


# In[ ]:


len(review_content)


# In[ ]:


len(rate)


# In[ ]:


review_title[:] = [titles.lstrip('\n') for titles in review_title]


# In[ ]:


review_title[:] = [titles.rstrip('\n') for titles in review_title]


# In[ ]:


review_content[:] = [titles.lstrip('\n') for titles in review_content]


# In[ ]:


review_content[:] = [titles.rstrip('\n') for titles in review_content]


# In[ ]:


import pandas as pd


# In[ ]:


df = pd.DataFrame()


# In[ ]:


df


# In[ ]:


df['Customer Name'] = cust_name


# In[ ]:


df['Review Title'] = review_title
df['Rating'] = rate
df['Reviews'] = review_content


# In[ ]:


df.head(5)


# In[ ]:


df.to_csv(r'E:RV1.csv',index = True)


# In[ ]:




