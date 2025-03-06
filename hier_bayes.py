import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mstats
import stan
import arviz as az
plt.style.use('ggplot')

df=pd.read_excel("data/multilevel_modeling.xlsx")
# print(df.head())
# print(df["id"].unique())
groups=df.groupby("id")
# plt.figure(figsize=(9,9))
# for name, group in groups:
#     plt.plot(group["age"], group["height"],  label=name)
# plt.legend()
# plt.xlabel("Age")
# plt.ylabel("Height")
# plt.show()
stan_model="""
data {
    int N;
    int N_id;
    array[N] int<lower=1,upper=N_id> id;
    vector[N] X;
    vector[N] Y;
}
parameters {    
    real alpha;
    real beta;
    vector[N_id] a_id;
    vector[N_id] b_id;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N_id] a;
    vector[N_id] b;
    for(i in 1:N_id){
        a[i]=alpha+a_id[i];
        b[i]=beta+b_id[i];
    }
}
model{
    for(i in 1:N_id){
        a_id[i]~normal(0,sigma_a);
        b_id[i]~normal(0,sigma_b);
    }
    for(i in 1:N){
        Y[i]~normal(a[id[i]]+b[id[i]]*X[i],sigma);
    }
}
"""
sm=stan.build(stan_model,data={"N":df.shape[0],"N_id":len(df["id"].unique()),"id":df["id"].values,"X":df["age"].values,"Y":df["height"].values})
fit=sm.sample(num_chains=3,num_samples=3000,num_warmup=500)
# with open("output.txt","w") as f:
#     print(az.summary(fit).to_string(),file=f)
# az.plot_trace(fit)
# plt.show()
ms_a=fit["a"].T
ms_b=fit["b"].T
print(ms_a.shape)
print(ms_b.shape)
x=np.arange(18)
df_b=pd.DataFrame([])
for i in range(x.shape[0]):
    df_b[i]=(ms_a[:,0]+ms_b[:,0]*x[i]).flatten()
low_y50,high_y50=mstats.mquantiles(df_b, [0.25,0.75], axis=0)
low_y95,high_y95=mstats.mquantiles(df_b, [0.025,0.975], axis=0)
df_0=groups.get_group(1)
print(df_0.head())
plt.plot(df_0["age"], df_0["height"])
plt.fill_between(x,low_y50,high_y50,alpha=0.6,color="darkgray")
plt.fill_between(x,low_y95,high_y95,alpha=0.3,color="gray")
plt.xlabel("Age")
plt.ylabel("Height")
plt.show()

