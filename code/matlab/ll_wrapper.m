function LL = ll_wrapper(xv,x0,fit_vec,data,model_type,dt,n,dimz,dimd,dimy,N,fr_func)

x = reparameterize(x0,dt,model_type,N,xv,fit_vec);
LL = LL_all_trials_v2(x,data,dt,n,dimz,dimd,dimy,model_type,N,fr_func,[]);

end