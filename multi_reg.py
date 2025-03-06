import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import stan
import seaborn as sns
import arviz as az
plt.style.use('ggplot')

df=pd.read_excel("data/real_estate2.xlsx")
# print(df.head())
df["elapsed"]=2018-df["year"]
# print(df["distance"].unique())
dis_array=df["distance"].unique()
dis_dict={dis_array[0]:10,dis_array[1]:15,dis_array[2]:5,dis_array[3]:20,dis_array[4]:30,dis_array[5]:np.nan}
print(dis_dict )
df["distance2"]=df["distance"].map(dis_dict)
print(df.head())
df=df.dropna()
df2=df[["space","elapsed","distance2","value"]]
print(df2.head())
# g = sns.PairGrid(df2)
# g=g.map_lower(sns.kdeplot)
# g=g.map_diag(sns.histplot, kde=False)
# g=g.map_upper(plt.scatter)
# # plt.show()
stan_model="""
data{
    int N;
    vector[N] space;
    vector[N] elapsed;
    vector[N] dis;
    vector[N] Y;
}
parameters{
    real a;
    real b1;
    real b2;
    real b3;
    real<lower=0> sigma;
}
model{
    vector[N] mu;
    for(i in 1:N){
        mu[i]=a+b1*space[i]+b2*elapsed[i]+b3*dis[i];
        Y[i]~normal(mu[i],sigma);
    }
}
"""
sm=stan.build(stan_model,data={"N":df2.shape[0],"space":df2["space"].values,"elapsed":df2["elapsed"].values,"dis":df2["distance2"].values,"Y":df2["value"].values})
fit=sm.sample(num_chains=3,num_samples=2000,num_warmup=500)
print(az.summary(fit))
az.plot_trace(fit,figsize=(10,20))
plt.savefig("multi_reg_trace.png")