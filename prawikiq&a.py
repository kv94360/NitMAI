#for importing the dataset
import tkinter as tk
import pandas as pd
def get_ans():
    text=que.get()
    val=qa_result(text)
    que.delete(0,tk.END)
    t.set(val)
window=tk.Tk()
window.geometry("800x300")
window.title("Wiki Q&A Chat Bot")

lab=tk.Label(window,text="Enter your question")
lab.grid(row=0,column=0)

t=tk.StringVar()
que=tk.Entry(window)
que.grid(row=0,column=1)

but=tk.Button(window,text="Get answer",command=get_ans)
but.grid(row=1,column=1)

lab2=tk.Label(window,textvariable=t,wraplength=250)
lab2.grid(row=2,column=0)



#reading the dataset
dataset=pd.read_csv('Book12.csv',encoding='latin-1')
convo_frame=dataset.iloc[:,[1,2]]
from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer(ngram_range=(1,3))
v=vect.fit_transform(convo_frame["Question"])
from sklearn.metrics.pairwise import cosine_similarity
def qa_result(q):
    ques=vect.transform([q])
    cs=cosine_similarity(ques,v)
    print(cs)
    rs=pd.Series(cs[0]).sort_values(ascending=False)
    print(rs)
    val=rs.iloc[0:1].values
    if val > 0:
        ans=rs.index[0]
        return convo_frame.iloc[ans]['Answer']
    else:
        return "We don't know the answer."

window.mainloop()