import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mstats
import stan
import arviz as az

plt.style.use('ggplot')

df=pd.read_excel("data/temperature_series.xlsx")
print(df.head())
# plt.plot(df["x"],df["y"])
# plt.show()
stan_model="""
data {
    int T;
    int T_new;
    vector[T] x;
    vector[T] y;
}
parameters {
    vector[T] mu;
    real<lower=0> s_Y;
    real<lower=0> s_mu;
    }
model {
    for(t in 2:T){
        mu[t]~normal(mu[t-1],s_mu);
    }
    for(t in 1:T){
        y[t]~normal(mu[t],s_Y);
    }
}
generated quantities {
    real mu_new;
    vector[T+T_new] y_new;
    for(t in 1:T){
        y_new[t]=normal_rng(mu[t],s_Y);
    }
    mu_new=normal_rng(mu[T],s_mu);
    y_new[T+T_new]=normal_rng(mu_new,s_Y);
}
"""
sm=stan.build(stan_model,data={"T":df.shape[0],"T_new":1,"x":df["x"].values,"y":df["y"].values})
fit=sm.sample(num_chains=3,num_samples=3000,num_warmup=500)
with open("output.txt","w") as f:
    print(az.summary(fit).to_string(),file=f)
# az.plot_trace(fit)
# plt.show()
y_new_array=fit["y_new"].T
low_y50,high_y50=mstats.mquantiles(y_new_array, [0.25, 0.75], axis=0)
low_y95,high_y95=mstats.mquantiles(y_new_array, [0.025, 0.975], axis=0)
plt.plot(df["x"],df["y"])
x=df["x"].values
x=np.append(x,2017)
plt.fill_between(x,low_y50,high_y50,alpha=0.5,color="darkgray")
plt.fill_between(x,low_y95,high_y95,alpha=0.2,color="gray")
plt.show()