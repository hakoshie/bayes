import pandas as pd
import matplotlib.pyplot as plt
import stan
import arviz as az

df=pd.read_excel("data/data.xlsx",index_col=0)
stan_model="""
data{
    int N;
    array[N] real Y;
}
parameters{
    real mu;
    real<lower=0> sigma;
}
model{
    for(n in 1:N){
        Y[n]~normal(mu,sigma);
    }
}   
"""

sm=stan.build(stan_model,data=dict(N=df.shape[0],Y=df[0].values))

fit=sm.sample(num_chains=1,num_samples=2000,num_warmup=500)    
# print(fit["mu"].mean())
# print(fit["sigma"].mean())
print(az.summary(fit))
data=az.from_pystan(posterior=fit,posterior_model=sm)
az.plot_trace(data,figsize=(12,6))
plt.savefig("trace.png")