import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import stan
import arviz as az
from scipy.stats import mstats
plt.style.use('ggplot')
df = pd.read_excel("data/real_estate1.xlsx")
print(df.head())

# plt.scatter(df["space"],df["value"])
# plt.show()
stan_model="""
data{
    int N;
    array[N] real space;
    array[N] real value;
    int N_s;
    array[N_s] real X_s;
    }
parameters{
    real alpha;
    real beta;
    real<lower=0> sigma;
    }
model{
    for(n in 1:N){
        value[n] ~ normal(alpha*space[n]+beta, sigma);
    }
}   
generated quantities{
    array[N_s] real Y_s;
    for(n in 1:N_s){
        Y_s[n] = normal_rng(alpha*X_s[n]+beta, sigma);
    }
    }
    """
# print(df["value"].values)
X_s=np.arange(40,90,1)
N_s=X_s.shape[0]
sm=stan.build(stan_model,data=dict(N=df.shape[0],space=df["space"].values,value=df["value"].values,N_s=N_s,X_s=X_s))
fit=sm.sample(num_chains=3,num_samples=2000,num_warmup=500)
print(az.summary(fit))

ms_a=fit["alpha"]
ms_b=fit["beta"]
df_b=pd.DataFrame([])
for i in range(40,90,1):
    df_b[i]=(ms_a*i+ms_b).flatten()
print(df_b.shape)
low_y50,high_y50=mstats.mquantiles(df_b,[0.25,0.75],axis=0)
low_y95,high_y95=mstats.mquantiles(df_b,[0.025,0.975],axis=0)
print(len(low_y50))
print(len(X_s))
plt.scatter(df["space"],df["value"])
plt.fill_between(X_s,low_y95,high_y95,alpha=0.6,color="gray")
plt.fill_between(X_s,low_y50,high_y50,alpha=0.6,color="darkgray")
a=78.2
b=-706.1
# x=np.arange(40,90,1)
y=a*X_s+b
plt.plot(X_s,y,color="black")
plt.scatter(df["space"],df["value"])
plt.show()

Y_p=fit["Y_s"]
Y_p=Y_p.T
low_y,high_y=mstats.mquantiles(Y_p,[0.025,0.975],axis=0)
plt.fill_between(X_s,low_y,high_y,alpha=0.6,color="gray")
y=a*X_s+b
plt.plot(X_s,y,color="black")
plt.scatter(df["space"],df["value"])
plt.show()