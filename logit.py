import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import stan
from scipy.stats import mstats
import arviz as az
plt.style.use('ggplot')

df=pd.read_excel("data/dose_response.xlsx")
print(df.head())
# plt.scatter(df["log10 C"],df["death"])
# plt.show()

stan_model="""
data{
    int N;
    vector[N] x;
    array[N]int<lower=0,upper=1> y;
}
parameters{
    real alpha;
    real beta;
}
model{
    for(n in 1:N){
        y[n]~bernoulli_logit(alpha+beta*x[n]);
    }
}
"""
sm=stan.build(stan_model,data=dict(N=df.shape[0],x=df["log10 C"].values,y=df["death"].values))
fit=sm.sample(num_chains=3,num_samples=2000,num_warmup=500)
print(az.summary(fit))
data=az.from_pystan(posterior=fit,posterior_model=sm)
az.plot_trace(data,figsize=(12,6))
plt.show()
a,b=fit["alpha"].mean(),fit["beta"].mean()
ms_a=fit["alpha"]
ms_b=fit["beta"]
x=np.arange(1.0,2.0,0.01)
f=lambda x:1/(1+np.exp(-x))
df_b=pd.DataFrame()
for i in range(x.shape[0]):
    df_b[x[i]]=f(ms_a+ms_b*x[i]).flatten()  
print(df_b.head())
low_y50,high_y50=mstats.mquantiles(df_b, [0.25, 0.75], axis=0)
low_y95,high_y95=mstats.mquantiles(df_b, [0.025, 0.975], axis=0)
plt.scatter(df["log10 C"],df["death"])
plt.plot(x,f(a+b*x),label="mean")
plt.fill_between(x,low_y50,high_y50,alpha=0.5,color="gray",label="50% CI")
plt.fill_between(x,low_y95,high_y95,alpha=0.5,color="darkgray",label="95% CI")
plt.legend()
plt.show()


