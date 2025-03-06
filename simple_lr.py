import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import stan
import arviz as az
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
    """
# print(df["value"].values)
sm=stan.build(stan_model,data=dict(N=df.shape[0],space=df["space"].values,value=df["value"].values))
fit=sm.sample(num_chains=3,num_samples=2000,num_warmup=500)
print(az.summary(fit))
# az.plot_trace(fit,figsize=(12,8))
# plt.show()
a=78.2
b=-706.1
x=np.arange(40,90,1)
y=a*x+b
plt.plot(x,y)
plt.scatter(df["space"],df["value"])
plt.show()