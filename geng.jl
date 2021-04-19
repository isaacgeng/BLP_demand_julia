### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ ba27bfd4-3174-40e2-a7f1-59ee10524890
using SparseArrays, Statistics, ForwardDiff, MAT, Optim, LinearAlgebra

# ╔═╡ 6bbe82e8-bc33-4a55-9a0f-c8b2362447f1
cd("/Users/isaacgeng/OneDrive - The Chinese University of Hong Kong/ECON5480-IO")

# ╔═╡ cfee6ce0-7a6a-4e8b-ba1e-e0f137a84a72
md"""
## 1. Data preparation
1.1 Read the input matrice.\
1.2 Calculate the miscelleous matrices for later use.
"""

# ╔═╡ 56d9b941-ee04-4003-9a19-5e1bd62fd1f2
begin	
	# load the files into matrixes, use meta-programming to reduce later
	ps2raw = matopen("data/ps2.mat")
	ivraw = matopen("data/iv.mat")
	x2 = read(ps2raw,  "x2")
	id = read(ps2raw, "id")
	s_jt = read(ps2raw, "s_jt")
	x1 = read(ps2raw, "x1")
	v = read(ps2raw,"v")
	demogr = read(ps2raw, "demogr")
	id_demo = read(ps2raw, "id_demo")
	iv = read(ivraw, "iv")
	close(ps2raw)
	close(ivraw)
end

# ╔═╡ fdce56e0-29ee-475e-a10e-22c452e3c64d
ns, nmkt, nbrn, n_inst = 20, 94, 24, 20

# ╔═╡ 9ce7689a-1afc-4e66-8bad-3ab1dca77da3
cdid = kron([1:nmkt]', ones(nbrn, 1))

# ╔═╡ 0da243dc-0f18-4e96-b8d6-b5d33a9fcc65
cdindex = 24:24:2256

# ╔═╡ 92f1eb9d-53b9-4910-86ff-ff2329e87031
IV = [iv[:,2:n_inst+1] x1[:, 2:nbrn+1]]

# ╔═╡ adbd8332-f16a-44ab-956b-45364ae2ff16
md"""
Starting values: 0 means corresponding coef not maxed over.
"""

# ╔═╡ bbdb3cd8-6023-444c-ab75-5789142aa626
θ_2w = [0.3772    3.0888         0    1.1859         0;
             1.8480   16.5980    -.6590         0   11.6245;
            -0.0035   -0.1925         0    0.0296         0;
             0.0810    1.4684         0   -1.5143         0]

# ╔═╡ 0eb95563-5a29-4ebb-a2fa-5bd70e37539e
ind = findall(!iszero,θ_2w)

# ╔═╡ 8e388be0-5958-4486-a689-b1c1c2b7883b
θ_2_init = filter(!iszero,θ_2w)

# ╔═╡ 6514513b-7bc1-42b0-b5b5-2f89db5a53f1
θ_i = getindex.(ind,1)

# ╔═╡ 3ca1524c-20bd-47c1-9578-3a943ad5d307
θ_j = getindex.(ind,2)

# ╔═╡ d40d1b80-8016-4be0-ab98-257c7b377b2a
md"""
create a weight matrix
"""

# ╔═╡ 38df8f14-9609-46c1-adbf-dc037ed2a081
IV'*IV

# ╔═╡ abea1050-a483-42f3-ba46-f37863d6a056
invA = inv(Matrix(IV'*IV))

# ╔═╡ 7ceb006b-9d2b-4392-8b29-68663ac9c3fc
md"""
## 2. Logit results and save the mean utility as initial values for the search below
"""

# ╔═╡ 1f14383f-1c32-4044-9545-abc473f0c80f
# compute the outside good market share by market
temp = cumsum(s_jt, dims=1)

# ╔═╡ 6bbc13da-2b84-4527-b89c-50d418f8bea0
sum1 = temp[cdindex,:] # total market share in each market besides outside option

# ╔═╡ c0625ee4-f0f4-4f28-8ba5-8d91d3bc04dd
sum1[2:size(sum1,1),:] = diff(sum1,dims=1)

# ╔═╡ 9e569165-a847-46fc-8004-da89d2858bb3
outshr = 1.0 .- repeat(sum1,inner=(24,1))

# ╔═╡ 64acc488-00e6-49cc-be70-55c7308d5627
y = log.(s_jt) - log.(outshr)

# ╔═╡ e97a0bb5-fb93-483b-931f-43e6e34fdd46
mid = x1'*IV*invA*IV'

# ╔═╡ 18f25b8c-fca6-45e2-9ba7-2179c2c7f313
t = (mid*x1) \ (mid*y)
# instead of using inv(mid*x1)*(mid*y), use this to utilize qr factorization.

# ╔═╡ bfa9baf8-a0cd-4c7c-a6b3-69c4d4f01c28
mvalold = exp.(x1*t) # fitted log shares

# ╔═╡ 9c93ef5a-7714-435c-a5ad-4fcf7fec049d
oldt2 = zeros(size(θ_2_init))

# ╔═╡ 817414ea-e30d-441e-8e0c-55dafcde1da1
vfull = repeat(v,inner=(24,1))

# ╔═╡ bf3d1cce-cd9c-465a-b3c2-37edec84da3b
dfull = repeat(demogr,inner=(24,1))

# ╔═╡ 2888cb4c-e29f-4b84-8858-98165d8fe531
md"""
## 3. The Random Coefficient Logit Demand Estimation
### 3.1 Function 1: individual choice probabilities
"""

# ╔═╡ 123da232-f975-40b5-8b27-39f33ae2cb0d
md"""
- original ind_sh matlab code
```
function f = ind_sh(expmval,expmu)
# This function computes the "individual" probabilities of choosing each brand
global ns cdindex cdid
eg = expmu.*kron(ones(1,ns),expmval);
temp = cumsum(eg); 
sum1 = temp(cdindex,:);
sum1(2:size(sum1,1),:) = diff(sum1);
denom1 = 1./(1+sum1);
denom = denom1(cdid,:);
f = eg.*denom;
```
"""

# ╔═╡ 417375fb-f62d-4415-a768-d4a3ecf042cf
md"""
- We first show the math of market share,  note that the integrad is individual choice probabilities.
$s_{j} = \int \frac{e^{\delta_j + x_j σ ν}}{1+\sum_{i=1}^J e^{\delta_i + x_i σ ν}} dF_ν(ν)$ 
- where \
$\delta_j = β[1]*p[j] + x[:,j]' *β[2:end] + ξ[j]$
- here, we choose to replicate the nevo's way, that is, we donot write out the formula $\delta_j$ in ind_sh function.
"""

# ╔═╡ 40d6d064-c96e-446e-8ec6-eff5be1bc4d7
function ind_sh(expmval, expmu)
	eg = expmu.*kron(ones(1,ns), expmval)
	# common techinique in nevo: begin
	temp = cumsum(eg,dims=1);
	sum1 = temp[cdindex,:]
	sum1[2:size(sum1, 1),:] = diff(sum1, dims=1)
	# end
	denom1 = 1 ./ (1 .+ sum1)
	# denom. = denom1[cdid,:]
	denom = repeat(denom1, inner=(24,1))
	f = eg.*denom
end

# ╔═╡ dafdd2b7-5992-420d-91e3-a0e64009db43
md"""
- the ind_sh function's second parameter `expmu` is the result of mufunc, we change the name to $\mu$ _func instead.
"""

# ╔═╡ 2d2d119b-bda2-4dc4-8364-0c93f5871bd9
function μ_func(x2,θ_2w)
	n,k = size(x2)
	j = size(θ_2w,2)-1
	μ = Matrix{Real}(undef, n,ns); # 2256, 20
	for i = 1:ns
    	v_i = vfull[:, i:ns:k*ns] # try v_i = vfull[:,3:20:80] if unclear
      	d_i = dfull[:, i:ns:j*ns]
 		μ[:,i] = x2.*v_i*θ_2w[:,1] + x2.*(d_i*θ_2w[:,2:j+1]')*ones(k,1);
	end
	f = μ;
end

# ╔═╡ 644f082d-51bc-4fda-b46b-ae0cfdffe3bb
md"""
- Note that here I use a different sentence to create a empty container where the type is implicitly specified as being subtypes of `Real`. This is needed since later on we use `Autodiff.jl` and `Autodiff.jl` uses dual to calculate its direvatives and Float64 type is not capable of obtaining its dual.
"""

# ╔═╡ 6cb4a7df-4670-4f60-9474-80f143c3671d
md"""
#### 3.1.1 debug and test: ind_sh
-  we test `ind_sh` work or not. And we found those julia syntax difference to matlab:
1. exp() in matlab need to be exp.() in case the matrix is not square
2. the square brackets near `x2.*v_i*θ_2w[:,1]` to remove
3. the square brackets near `d_i*θ_2w[:,2:j+1]'` should be round brackets. 
"""

# ╔═╡ 8f3e129d-a68f-433d-99e9-15d7ac2e4490
md"""
- debug console, for record only.
```
exp.(μ_func(x2,θ_2w))
size(x2.*v_i*θ_2w[:,1])
v_i = vfull[:,3:20:80]
```
"""

# ╔═╡ 9a226183-0b7b-448a-986b-722998900181
md"""
- test for $\mu$_func: passed
"""

# ╔═╡ 31a69149-b3df-4f45-a6e4-0209cf0aaccb
μ_func(x2,θ_2w)

# ╔═╡ dcb6c090-f3da-447b-802b-23c8d4881a33
md"""
- test for ind_sh func: passed
"""

# ╔═╡ 8c5e09da-baaf-4b9c-a90d-b70d5130d47d
a, b = mvalold, exp.(μ_func(x2,θ_2w))

# ╔═╡ 6f1d4752-763a-415d-9b54-d12807ebb574
ind_sh(a, b)

# ╔═╡ 4ee9e2b7-ab19-46f3-9817-0b35728bbe19
md"""
### 3.2 Market share function
- original matlab code
```
function f = mktsh(mval, expmu)
% This function computes the market share for each product
% Written by Aviv Nevo, May 1998.
global ns 
f = sum((ind_sh(mval,expmu))')/ns;
f = f';
```
"""

# ╔═╡ 5f690e8f-a375-40c3-8a52-6629f04826e5
function mktsh(mval, expmu)
	f = sum(ind_sh(mval, expmu)', dims=1)./ns;
	f = f'
end

# ╔═╡ 8a29a56b-c217-406e-a8f3-3452913b4deb
md"""
#### 3.2.1 test for mktsh: passed
"""

# ╔═╡ 9507da3c-d833-4bb1-a448-eb4d2da76a1c
mktsh(a, b)

# ╔═╡ 16fb264a-ed75-4d57-9cd2-610499653c9b
md"""
### 3.3 Mean utility level function
"""

# ╔═╡ b37444a4-4296-4d8f-81c9-e428d9b2d9a4
function meanval(θ_2, oldt2=oldt2, mvalold=mvalold)
	if maximum(abs.(θ_2 - oldt2)) < 0.01
		tol = 1e-9
		flag = 0
	else
		tol = 1e-6
		flag = 1
	end
	
	θ_2w = Matrix(sparse(θ_i,θ_j, θ_2))
	expmu = exp.(μ_func(x2, θ_2w))
	norm = 1
	avgnorm = 1
	
	i = 0
	
	# contraction mapping to compute mean utility
	while (norm > tol*10^(flag*floor(i/50))) & (avgnorm > 1e-3*tol*10^(flag*floor(i/50)))
		mval = mvalold .* s_jt ./ mktsh(mvalold, expmu);
		t = abs.(mval - mvalold)
		norm = maximum(t)
		avgnorm = mean(t)
		mvalold = mval
		i = i + 1
	end
	
	println("# of iterations. for delta convergence: "*string(i))
	
	if flag == 1 & maximum(isnan.(mval)) < 1
		mval = mvalold .* s_jt ./ mktsh(mvalold, expmu)
		mvalold = mval
		oldt2 = θ_2
	end
	
	log.(mval)
end

# ╔═╡ a2b63874-1949-4bb5-85e1-30cde5795c61
md"""
#### 3.3.1 Test for meanval: passed
"""

# ╔═╡ 6f69fd00-2db3-4b5e-a261-b45d7b04cc8b
meanval(θ_2_init)

# ╔═╡ f4bf30cd-4829-4d18-8cee-844defcbc26a
md"""
#### 3.3.2 (`Q3.3`) Write a routine that saves `avgnorm, norm`, graph and discuss.
"""

# ╔═╡ 7d3f24b3-aedc-4118-927f-87ae54ceefc0
md"""
### 3.4 Jacobian of Mval and Gmmobjg
- I utilize auto-differentiation to save me from the approx calculation process.
"""

# ╔═╡ 06900cab-44da-4d1d-9fab-11ef3d67b3bc
jacob = θ_2 -> ForwardDiff.jacobian(x -> meanval(x), θ_2)

# ╔═╡ 2487fa3d-ce0e-48e0-bada-2698d0c4b881
function jacob!(g,θ_2)
	g = Matrix{Real}(undef, nbrn*nmkt,size(θ_2)[1])
	g = jacob(θ_2)
	return g
end

# ╔═╡ 9019b1a3-8235-4d46-b9ea-6e9fe0a37dd3
md"""
#### 3.4.1 Test for jacob and jacob! function: passed
"""

# ╔═╡ 620ff1f8-b67d-4e8f-93ee-c7a8938e9769
g = Matrix{Real}(undef, 2256,13);

# ╔═╡ 0f1f6bea-091a-460b-8cec-86d577e1c463
jacob(θ_2_init)

# ╔═╡ fe5b09ff-1eea-46c0-8779-b4b4cdfebe7e
jacob!(g, θ_2_init)

# ╔═╡ 03fa5b43-a49f-439c-9f6b-830ddbd750a5
md"""
#### 3.4.2 Comparing the autodiff with matlab result.
"""

# ╔═╡ d3f2e237-2264-46ae-a2d8-59b587c396d2
md"""
- I compared the results of hand written approximated jacobian (**matlab result below**) and found that there is some degree of difference between the audodiff result and the jacobian approximation result.
"""

# ╔═╡ 7508e049-acb9-4761-8c7f-4f2d8383adb2
begin
	jcb_approx_raw = matopen("data/jcb.mat")
	jcb_approx = read(jcb_approx_raw,"jcb")
	close(jcb_approx_raw)
end

# ╔═╡ d68eb579-b818-42f0-b9b4-ad970bfb55e2
jcb_approx

# ╔═╡ b90497b6-0020-487d-9a6e-5047eaf7e8cd
md"""
### 3.5 GMM objective function: gmmobjg
"""

# ╔═╡ ad667085-b7a9-4ec3-8a7e-e2670a3072f5
md"""
- I manually splitted the function into three functions and the code duplcation is hard to avoid due to that `gmmobjg` need to be single valued function to be optimized by using `Optim.jl` package.
"""

# ╔═╡ c3d3c28d-2458-4c91-b346-086fd4f77453
function gmmobjg(θ_2,nargout=0)
	δ = meanval(θ_2)
	# deals with cases were the min algorithm drifts into region where the objective is not defined
	if maximum(isnan.(δ)) == 1
		f = 1e+10
	else
		temp1 = x1' * IV
		temp2 = δ' * IV
		θ_1= (temp1 * invA * temp1') \ (temp1 * invA * temp2')
		gmmresid = δ - x1*θ_1
		temp1 = gmmresid' * IV;
		f1 = temp1 * invA * temp1';
		f = f1
		# donot know what is for 
		if nargout > 1
			temp = jacob(mvalold,θ_2)'
			df = 2 * temp * IV * invA * IV' * gmmresid
		end
	end
	# if nargout > 1
	# 	f, df, gmmresid
	# else
	# 	f, gmmresid
	# end
	real(f[1,1])
end
		

# ╔═╡ 497fc8fb-6c26-4b44-88f3-62df000c05ae
function gmmresid(θ_2, nargout=0)
	δ = meanval(θ_2)
	# deals with cases were the min algorithm drifts into region where the objective is not defined
	if maximum(isnan.(δ)) == 1
		f = 1e+10
	else
		temp1 = x1' * IV
		temp2 = δ' * IV
		θ_1= (temp1 * invA * temp1') \ (temp1 * invA * temp2')
		gmmresid = δ - x1*θ_1
		temp1 = gmmresid' * IV;
		f1 = temp1 * invA * temp1';
		f = f1
		# donot know what is for 
		if nargout > 1
			temp = jacob(mvalold,θ_2)'
			df = 2 * temp * IV * invA * IV' * gmmresid
		end
	end
	gmmresid
end
	

# ╔═╡ 8395e98e-197f-423d-83fc-66a6bcc80afc
function θ_1(θ_2, nargout=0)
	δ = meanval(θ_2)
	# deals with cases were the min algorithm drifts into region where the objective is not defined
	if maximum(isnan.(δ)) == 1
		f = 1e+10
	else
		temp1 = x1' * IV
		temp2 = δ' * IV
		θ_1= (temp1 * invA * temp1') \ (temp1 * invA * temp2')
		gmmresid = δ - x1*θ_1
		temp1 = gmmresid' * IV;
		f1 = temp1 * invA * temp1';
		f = f1
		# donot know what is for 
		if nargout > 1
			temp = jacob(mvalold,θ_2)'
			df = 2 * temp * IV * invA * IV' * gmmresid
		end
	end
	θ_1
end

# ╔═╡ d71340f0-6bfa-42d9-b3f8-819ef536eccb
md"""
#### 3.5.1 Test for gmmobjg, gmmresid, $\theta$_1: passed.
"""

# ╔═╡ 1b56809a-656a-41c2-8164-85d532561565
gmmobjg(θ_2_init)

# ╔═╡ b1fb516d-2368-4c85-b774-d12bd12992f2
gmmresid(θ_2_init)

# ╔═╡ 7757a089-7824-4924-b31f-8788d412cc00
ForwardDiff.gradient(gmmobjg,θ_2_init)

# ╔═╡ c3ca972d-77bc-4b44-94cb-f32100d15c8d
md"""
#### 3.5.2 Jacobians wrt gmmobjg
- later I showed that using Jacobian wrt mval or gmmobjg gives you exact same result.
"""

# ╔═╡ 6a7ec604-b739-48dc-a9a9-287cddf3f6c9
function jcb!(g,θ_2)
	g = Matrix{Real}(undef, 1,size(θ_2)[1])
	g = ForwardDiff.gradient(gmmobjg,θ_2)
	g
end

# ╔═╡ fbed38dc-7535-4c32-92d9-f52ef6cb4073
md"""
### 3.6 Variance-covariance matrix of gmmobjg func wrt $\theta_2$
"""

# ╔═╡ 26c81b54-7760-4eed-96c4-f9a9f2f4b532
function var_cov(θ_2,	gmmresid=gmmresid(θ_2))
	N = size(x1,1)
	Z = size(IV,2)
	temp = jacob(θ_2)
	a = [x1 temp]' * IV
	IVres = IV.*(gmmresid*ones(1, Z))
	b = IVres' * IVres
	f = inv(a*invA*a')*a*invA*b*invA*a'*inv(a*invA*a')
end

# ╔═╡ b96160bb-ffb8-43bf-8e7e-2e4dd221b0ee
md"""
#### 3.6.1 Test for var_cov: passed.
"""

# ╔═╡ 9b11c994-4892-4648-b9e1-6fb069a88304
var_cov(θ_2_init)

# ╔═╡ d10bf777-a869-4e99-aa6b-8d9327dfad91
md"""
## 4. Final routine solving for optimal $\theta_2$ to gmmobjg
"""

# ╔═╡ 2040ca28-08ea-4e29-b9a5-f0276122bf78
md"""
- I found that using derivatives on mealval function or on gmmobjg itself has no difference as below shows.
"""

# ╔═╡ d22d94df-1498-439d-8bcf-a5e2379ab70a
Optim.minimizer(optimize(gmmobjg, jacob!,θ_2_init, BFGS()))

# ╔═╡ ca002c77-5a15-4bb7-a726-d02625317ab9
θ2 = Optim.minimizer(optimize(gmmobjg, jcb!,θ_2_init, BFGS()))

# ╔═╡ c8cd2a43-8ece-4e26-a5d4-14630d6572d6
md"""
- extract execution time and final objective function value.
"""

# ╔═╡ 5385600c-5ad4-48eb-beed-a430e5ee788c
ttime = @elapsed res = optimize(gmmobjg, jcb!,θ_2_init, BFGS())

# ╔═╡ 0841ae53-f235-46a0-8b7d-83843898d7c0
fval = Optim.minimum(res)

# ╔═╡ ee1e5433-8dbc-4e62-b667-36ce3a1cb763
begin
	vcov = var_cov(θ2)
	se = real(sqrt(var_cov(θ2)))
end

# ╔═╡ b3328c91-e6f2-4c38-b8a0-75ebe306ca82
begin
	θ2w = Matrix(sparse(θ_i,θ_j,θ2))
	t_new = size(se,1) - size(θ2,1)
	se2w = Matrix(sparse(θ_i, θ_j, se[t_new+1:size(se,1)]))
end

# ╔═╡ 108fa10b-e075-430a-bbfe-b15d1f8d31e3
θ1 = θ_1(θ2)

# ╔═╡ 4e441ea2-1ae9-4052-a238-cf227674614a
begin
	Ω = inv(vcov[2:25,2:25])
	xmd = [x2[1:24,1] x2[1:24,3:4]];
	ymd = θ1[2:25];
	β = (xmd' * Ω * xmd) \ (xmd' * Ω * ymd)
	resmd = ymd - xmd * β
	semd = sqrt.(diag(inv(xmd' * Ω * xmd)))
	mcoef = [β[1]; θ1[1]; β[2:3]];
	semcoef = [semd[1]; se[1]; semd]
end

# ╔═╡ 0b52131c-9488-4175-8572-646719a9a8cb
Rsq = 1 - ((resmd .- mean(resmd))'*(resmd .- mean(resmd)))/ ((ymd .- mean(ymd))'*(ymd .-mean(ymd)))

# ╔═╡ b17c094c-d231-45d6-8bf3-c30d178ff732
Rsq_G = 1-(resmd'*Ω*resmd)/((ymd .- mean(ymd))'*Ω*(ymd .- mean(ymd)))

# ╔═╡ 388ffa99-ec8e-465b-8355-31508de951de
Chisq = size(id,1)*resmd'*Ω*resmd

# ╔═╡ 1a042b8b-2c67-492b-9b86-3b1276a6616f
mcoef1, θ2w1, semcoef1, se2w1 = round.(mcoef,digits=4), round.(θ2w,digits=4), round.(semcoef,digits=4), round.(se2w, digits=4);

# ╔═╡ 473f2f56-4176-4e20-b36a-0d03d8c4c7b5
md"""
#### 4.1 Show the results and comparison to nevo's
"""

# ╔═╡ 3dd0dfeb-928e-45fe-aff3-8c42dae39cd0
s = 
"
	       Mean   Sigma  Income Income^2 Age Child \n
Constant: $(vcat(mcoef1[1], θ2w1[1,:]))\n
Constant se: $(vcat(semcoef1[1], se2w1[1,:]))\n
Price: $(vcat(mcoef1[2], θ2w1[2,:]))\n
Price se: $(vcat(semcoef1[1], se2w1[1,:]))\n
Sugar: $(vcat(mcoef1[3], θ2w1[3,:]))\n
Sugar se: $(vcat(semcoef1[3], se2w1[3,:]))\n
Mushy: $(vcat(mcoef1[4], θ2w1[4,:]))\n
Mushy se: $(vcat(semcoef1[4], se2w1[4,:]))\n
GMM Objective: $fval \n
MR R-squared: $Rsq \n
MR Weighted R-squared: $Rsq_G \n
run time: $ttime seconds
";


# ╔═╡ eb8da8fa-1984-4eda-a947-77e4da924c39
Text(s) # show the table of results

# ╔═╡ 4001b7e2-fde2-4d79-ad20-35b6b433b736
md"""
The result is really close to matlab's results wrt estimates. 
However, the ses especially the one wrt price is a lot difference.
My estimation of se is generally **smaller** than Nevo's, indicating the benefits of using a forward difference algorithm by `autodiff.jl`.
"""

# ╔═╡ 587f4cbb-d0cb-41ef-b95a-0049bcaa09d4
"
constant  
   -1.8866    0.3772    3.0888         0    1.1859         0

   (1,1)       0.2476
   (1,2)       0.1358
   (1,3)       1.1939
   (1,5)       1.0194

price     
  -32.4374    1.8480   16.5980   -0.6590         0   11.6245

   (1,1)       7.5809
   (1,2)       1.0913
   (1,3)     173.9858
   (1,4)       9.0578
   (1,6)       5.3098

sugar     
    0.1576   -0.0035   -0.1925         0    0.0296         0

   (1,1)       0.2476
   (1,2)       0.0129
   (1,3)       0.0466
   (1,5)       0.0374

mushy     
    0.9842    0.0810    1.4684         0   -1.5143         0

   (1,1)       0.0132
   (1,2)       0.2078
   (1,3)       0.6975
   (1,5)       1.1095

";

# ╔═╡ 347c9e65-ba7f-44ce-959e-838466abb44d
md"""
### 5. Conclusion
- This marks the end of the file.
"""

# ╔═╡ Cell order:
# ╠═ba27bfd4-3174-40e2-a7f1-59ee10524890
# ╠═6bbe82e8-bc33-4a55-9a0f-c8b2362447f1
# ╟─cfee6ce0-7a6a-4e8b-ba1e-e0f137a84a72
# ╠═56d9b941-ee04-4003-9a19-5e1bd62fd1f2
# ╠═fdce56e0-29ee-475e-a10e-22c452e3c64d
# ╠═9ce7689a-1afc-4e66-8bad-3ab1dca77da3
# ╠═0da243dc-0f18-4e96-b8d6-b5d33a9fcc65
# ╠═92f1eb9d-53b9-4910-86ff-ff2329e87031
# ╟─adbd8332-f16a-44ab-956b-45364ae2ff16
# ╠═bbdb3cd8-6023-444c-ab75-5789142aa626
# ╠═0eb95563-5a29-4ebb-a2fa-5bd70e37539e
# ╠═8e388be0-5958-4486-a689-b1c1c2b7883b
# ╠═6514513b-7bc1-42b0-b5b5-2f89db5a53f1
# ╠═3ca1524c-20bd-47c1-9578-3a943ad5d307
# ╟─d40d1b80-8016-4be0-ab98-257c7b377b2a
# ╠═38df8f14-9609-46c1-adbf-dc037ed2a081
# ╠═abea1050-a483-42f3-ba46-f37863d6a056
# ╠═7ceb006b-9d2b-4392-8b29-68663ac9c3fc
# ╠═1f14383f-1c32-4044-9545-abc473f0c80f
# ╠═6bbc13da-2b84-4527-b89c-50d418f8bea0
# ╠═c0625ee4-f0f4-4f28-8ba5-8d91d3bc04dd
# ╠═9e569165-a847-46fc-8004-da89d2858bb3
# ╠═64acc488-00e6-49cc-be70-55c7308d5627
# ╠═e97a0bb5-fb93-483b-931f-43e6e34fdd46
# ╠═18f25b8c-fca6-45e2-9ba7-2179c2c7f313
# ╠═bfa9baf8-a0cd-4c7c-a6b3-69c4d4f01c28
# ╠═9c93ef5a-7714-435c-a5ad-4fcf7fec049d
# ╠═817414ea-e30d-441e-8e0c-55dafcde1da1
# ╠═bf3d1cce-cd9c-465a-b3c2-37edec84da3b
# ╟─2888cb4c-e29f-4b84-8858-98165d8fe531
# ╟─123da232-f975-40b5-8b27-39f33ae2cb0d
# ╟─417375fb-f62d-4415-a768-d4a3ecf042cf
# ╠═40d6d064-c96e-446e-8ec6-eff5be1bc4d7
# ╟─dafdd2b7-5992-420d-91e3-a0e64009db43
# ╠═2d2d119b-bda2-4dc4-8364-0c93f5871bd9
# ╟─644f082d-51bc-4fda-b46b-ae0cfdffe3bb
# ╟─6cb4a7df-4670-4f60-9474-80f143c3671d
# ╟─8f3e129d-a68f-433d-99e9-15d7ac2e4490
# ╟─9a226183-0b7b-448a-986b-722998900181
# ╠═31a69149-b3df-4f45-a6e4-0209cf0aaccb
# ╟─dcb6c090-f3da-447b-802b-23c8d4881a33
# ╠═8c5e09da-baaf-4b9c-a90d-b70d5130d47d
# ╠═6f1d4752-763a-415d-9b54-d12807ebb574
# ╟─4ee9e2b7-ab19-46f3-9817-0b35728bbe19
# ╠═5f690e8f-a375-40c3-8a52-6629f04826e5
# ╟─8a29a56b-c217-406e-a8f3-3452913b4deb
# ╠═9507da3c-d833-4bb1-a448-eb4d2da76a1c
# ╟─16fb264a-ed75-4d57-9cd2-610499653c9b
# ╠═b37444a4-4296-4d8f-81c9-e428d9b2d9a4
# ╟─a2b63874-1949-4bb5-85e1-30cde5795c61
# ╠═6f69fd00-2db3-4b5e-a261-b45d7b04cc8b
# ╟─f4bf30cd-4829-4d18-8cee-844defcbc26a
# ╟─7d3f24b3-aedc-4118-927f-87ae54ceefc0
# ╠═06900cab-44da-4d1d-9fab-11ef3d67b3bc
# ╠═2487fa3d-ce0e-48e0-bada-2698d0c4b881
# ╠═9019b1a3-8235-4d46-b9ea-6e9fe0a37dd3
# ╠═620ff1f8-b67d-4e8f-93ee-c7a8938e9769
# ╠═0f1f6bea-091a-460b-8cec-86d577e1c463
# ╠═fe5b09ff-1eea-46c0-8779-b4b4cdfebe7e
# ╟─03fa5b43-a49f-439c-9f6b-830ddbd750a5
# ╟─d3f2e237-2264-46ae-a2d8-59b587c396d2
# ╠═7508e049-acb9-4761-8c7f-4f2d8383adb2
# ╠═d68eb579-b818-42f0-b9b4-ad970bfb55e2
# ╠═b90497b6-0020-487d-9a6e-5047eaf7e8cd
# ╟─ad667085-b7a9-4ec3-8a7e-e2670a3072f5
# ╠═c3d3c28d-2458-4c91-b346-086fd4f77453
# ╠═497fc8fb-6c26-4b44-88f3-62df000c05ae
# ╠═8395e98e-197f-423d-83fc-66a6bcc80afc
# ╟─d71340f0-6bfa-42d9-b3f8-819ef536eccb
# ╠═1b56809a-656a-41c2-8164-85d532561565
# ╠═b1fb516d-2368-4c85-b774-d12bd12992f2
# ╠═7757a089-7824-4924-b31f-8788d412cc00
# ╟─c3ca972d-77bc-4b44-94cb-f32100d15c8d
# ╠═6a7ec604-b739-48dc-a9a9-287cddf3f6c9
# ╠═fbed38dc-7535-4c32-92d9-f52ef6cb4073
# ╠═26c81b54-7760-4eed-96c4-f9a9f2f4b532
# ╟─b96160bb-ffb8-43bf-8e7e-2e4dd221b0ee
# ╠═9b11c994-4892-4648-b9e1-6fb069a88304
# ╟─d10bf777-a869-4e99-aa6b-8d9327dfad91
# ╟─2040ca28-08ea-4e29-b9a5-f0276122bf78
# ╠═d22d94df-1498-439d-8bcf-a5e2379ab70a
# ╠═ca002c77-5a15-4bb7-a726-d02625317ab9
# ╟─c8cd2a43-8ece-4e26-a5d4-14630d6572d6
# ╠═5385600c-5ad4-48eb-beed-a430e5ee788c
# ╠═0841ae53-f235-46a0-8b7d-83843898d7c0
# ╠═ee1e5433-8dbc-4e62-b667-36ce3a1cb763
# ╠═b3328c91-e6f2-4c38-b8a0-75ebe306ca82
# ╠═108fa10b-e075-430a-bbfe-b15d1f8d31e3
# ╠═4e441ea2-1ae9-4052-a238-cf227674614a
# ╠═0b52131c-9488-4175-8572-646719a9a8cb
# ╠═b17c094c-d231-45d6-8bf3-c30d178ff732
# ╠═388ffa99-ec8e-465b-8355-31508de951de
# ╠═1a042b8b-2c67-492b-9b86-3b1276a6616f
# ╟─473f2f56-4176-4e20-b36a-0d03d8c4c7b5
# ╟─3dd0dfeb-928e-45fe-aff3-8c42dae39cd0
# ╠═eb8da8fa-1984-4eda-a947-77e4da924c39
# ╟─4001b7e2-fde2-4d79-ad20-35b6b433b736
# ╠═587f4cbb-d0cb-41ef-b95a-0049bcaa09d4
# ╟─347c9e65-ba7f-44ce-959e-838466abb44d
