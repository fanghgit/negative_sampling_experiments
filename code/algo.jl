import StatsBase.sample
function softmax_greedy(X, Y, stepsize, W, n_epoch, topk, exact=1)
    # SGD
    N, D = size(X)
    K = size(Y, 1)
    
    #n_epoch = 50
    #W = zeros(D, K)
    #tic()
    #XW = X*W
    #@show(toc())
    XT = X'
    for epoch = 1:n_epoch
        #grad = zeros(D, K)
        #XW = X*W
        
        perm = randperm(N)
        tic()
        prec = 0.0
        prec_rand = 0.0
        for i = 1:1000
            grad = zeros(D, K)
            sample_idx = perm[i]
            class = find(Y[:,sample_idx])[1]
            Wx = W'*XT[:,sample_idx]
            #get topk
            if exact == 1
                #exact topk
                top_idx = sortperm(Wx, rev=true)[1:topk]
            elseif exact == 2
                # greedy mips topk
                maxWx = zeros(K)
                for j in 1:K
                    maxWx[j] = maximum( W[:,j] .* XT[:,sample_idx] )
                end
                top_idx = sortperm(maxWx, rev=true)[1:topk]
            else
                # random topk
                top_idx = sample([1:K;], topk)
            end
            
            
            # check greedy mips
            maxWx = zeros(K)
            for j in 1:K
                maxWx[j] = maximum( W[:,j] .* XT[:,sample_idx] )
            end
            top_idx_gmips = sortperm(maxWx, rev=true)[1:topk]
            top_idx_true = sortperm(Wx, rev=true)[1:topk]
            top_idx_rand = sample([1:K;], topk)
            #println( sortperm(Wx, rev=true) )
            #if i == 500
                #@show(top_idx_gmips)
                #@show(top_idx_true)
                #@show( intersect(top_idx_gmips, top_idx_true) )
            #end
            prec += length(intersect(top_idx_gmips, top_idx_true[1:10]) )/10
            prec_rand +=length(intersect(top_idx_rand, top_idx_true[1:10]) )/10
            
            for kk in 1:topk
                if class == top_idx[kk]
                    deleteat!(top_idx, kk)
                    break
                end
            end
            
            noise_sample = []
            while length(noise_sample) < 10
                noise = rand(1:K)
                if noise in top_idx || noise == class
                    continue
                else
                    append!(noise_sample, noise)
                end
            end
            
            tmp = exp.(Wx[top_idx])
            deno_estimate = sum(tmp) + exp(Wx[class]) + (K - length(top_idx) - 1)/length(noise_sample) * sum( exp.(Wx[noise_sample]) )
            
            # for j = 1:K
            #     grad[:,j] = tmp[j]/deno_estimate * XT[:,sample_idx]
            # end
            # grad[:, class] -= XT[:,sample_idx]
            
                    
            #update
            for ii in top_idx
                W[:,ii] -= stepsize* exp(Wx[ii])/deno_estimate * XT[:,sample_idx]
            end
            
            W[:,class] -= stepsize*(exp(Wx[class])/deno_estimate - 1) * XT[:,sample_idx]
            
        end
        @show(toq())
        @printf "%2.1e epoch complete \n" epoch
        
        # calculate loss
        prec = prec / 1000
        prec_rand = prec_rand / 1000
        @printf "prec@10 for gmips: %2.6e, prec@10 for random: %2.6e \n" prec prec_rand
        loss = 0
        risk = 0.0
        XW = X*W
        for i = 1:N
            Wxi = XW[i,:]
            class = find(Y[:,i])[1]
            deno = sum( exp.(Wxi) )
            loss += -log( exp(Wxi[class]) / deno )
            
            ii = indmax(Wxi)
            if ii != class
                risk += 1 
            end
        end
        loss = loss / N
        loss = loss / log(2)
        risk = risk / N
        @printf "loss: %3.3e, risk %3.3e \n" loss risk
        
    end
    return W
end
    

# softmax
function softmax(X, Y, stepsize, W, n_epoch, topk)
    # SGD
    N, D = size(X)
    K = size(Y, 1)
    
    #n_epoch = 50
    #W = zeros(D, K)
    #tic()
    #XW = X*W
    #@show(toc())
    XT = X'
    for epoch = 1:n_epoch
        #grad = zeros(D, K)
        #XW = X*W
        
        perm = randperm(N)
        tic()
        for i = 1:1000
            #println(i)
            grad = zeros(D, K)
            sample_idx = perm[i]
            class = find(Y[:,sample_idx])[1]
            Wx = W'*XT[:,sample_idx]
            tmp = exp.(Wx)
            deno = sum(tmp)
            for j = 1:K
                grad[:,j] = tmp[j]/deno * XT[:,sample_idx]
            end
            grad[:, class] -= XT[:,sample_idx]
            
            #update
            W -= stepsize*grad
        end
        @show(toq())
        @printf "%2.1e epoch complete \n" epoch
        
        # show topk statistics
        # randomly choose 3 samples
        for i = 1:5
            sample_idx = rand(1:N)
            class = find(Y[:,sample_idx])[1]
            Wx = W'*XT[:,sample_idx]
            tmp = exp.(Wx)
            grad = zeros(D, K)
            deno = sum(tmp)
            for j = 1:K
                grad[:,j] = tmp[j]/deno * XT[:,sample_idx]
            end
            grad[:, class] -= XT[:,sample_idx]
            
            norm_grad = zeros(K)
            for j = 1:K
                norm_grad[j] = norm( grad[:,j], 1 ) 
            end
            
            
            sorted_tmp = sort(norm_grad, rev=true)
            total = sum(sorted_tmp)
            total_topk = sum(sorted_tmp[1:topk])
            true_tmp = norm_grad[class]
            # @show(norm_grad[class])
            # @show(sum(norm_grad))
            # @printf "    top %i propotion: %1.4e\n" topk total_topk/total
            # @printf "    true class propotion: %1.4e\n" true_tmp/total
        end
        
        # calculate loss
        loss = 0
        risk = 0.0
        XW = X*W
        for i = 1:N
            Wxi = XW[i,:]
            class = find(Y[:,i])[1]
            deno = sum( exp.(Wxi) )
            loss += -log( exp(Wxi[class]) / deno )
            
            ii = indmax(Wxi)
            if ii != class
                risk += 1 
            end
        end
        loss = loss / N
        loss = loss / log(2)
        risk = risk / N
        @printf "loss: %3.3e, risk %3.3e \n" loss risk
        
    end
    return W
end

# pairwise rank loss
function train_pairwise(X, Y, stepsize, n_epoch, W)
    N, D = size(X)
    K = size(Y, 1)
    
    #W = zeros(D, K)
    XT = X'
    for epoch in 1:n_epoch
        perm = randperm(N)
        tic()
        for i = 1:N
            rand_class = rand(1:K)
            sample_idx = perm[i]
            class = find(Y[:,sample_idx])[1]
            if rand_class == class
                continue
            end
            tmp_class = dot(W[:,class], XT[:, sample_idx])
            tmp_other = dot(W[:,rand_class], XT[:, sample_idx])
            grad = exp(tmp_other - tmp_class) / (1 + exp(tmp_other - tmp_class)) * XT[:, sample_idx]
            
            # update
            W[:, class] += stepsize * grad
            W[:, rand_class] -= stepsize * grad
            
        end
        
        @show(toq())
        @printf "%2.1e epoch complete \n" epoch
        
        # calculate loss
        loss = 0
        risk = 0.0
        XW = X*W
        for i = 1:N
            Wxi = XW[i,:]
            class = find(Y[:,i])[1]
            #deno = sum( exp.(Wxi) )
            #loss += -log( exp(Wxi[class]) / deno )
            tmp_class = dot(W[:,class], XT[:, i])
            #for j = 1:K
            #    if j == class
            #        continue
            #    end
            #    tmp_other = XW[i,j]
            #    loss += log( 1 + exp(tmp_other - tmp_class ) )
            #end
            
            ii = indmax(Wxi)
            if ii != class
                risk += 1 
            end
        end
        loss = loss / N
        loss = loss / log(2)
        risk = risk / N
        @printf "loss: %3.3e, risk %3.3e \n" loss risk
        
    end
    
    return W
end


import StatsBase.sample
# negative sampling
function train_neg_sampling(X, Y, stepsize, n_epoch, W, k)
    N, D = size(X)
    K = size(Y, 1)
    
    #W = zeros(D, K)
    XT = X'
    # set # neg samples to be 5
    #k = 5
    for epoch in 1:n_epoch
        perm = randperm(N)
        tic()
        for i = 1:N
            sample_idx = perm[i]
            # choose negative samples
            class = find(Y[:,sample_idx])[1]
            neg_samples = sample([1:K;], k)
            grad = 1 / (1 + exp( dot(W[:,class], XT[:,sample_idx]) )) * XT[:, sample_idx]
            # update
            W[:,class] += stepsize*grad
            for ss in neg_samples
                if ss == class
                    continue
                end
                grad = 1 / (1 + exp(-dot(W[:,ss], XT[:,sample_idx]))) * XT[:, sample_idx]
                # update
                W[:,ss] -= stepsize*grad
            end
            
            
        end
        
        @show(toq())
        @printf "%2.1e epoch complete \n" epoch
        
        # calculate loss
        loss = 0
        risk = 0.0
        XW = X*W
        for i = 1:N
            Wxi = XW[i,:]
            class = find(Y[:,i])[1]
            #deno = sum( exp.(Wxi) )
            #loss += -log( exp(Wxi[class]) / deno )
            tmp_class = dot(W[:,class], XT[:, i])
            #for j = 1:K
            #    if j == class
            #        continue
            #    end
            #    tmp_other = XW[i,j]
            #    loss += log( 1 + exp(tmp_other - tmp_class ) )
            #end
            
            ii = indmax(Wxi)
            if ii != class
                risk += 1 
            end
        end
        loss = loss / N
        loss = loss / log(2)
        risk = risk / N
        @printf "loss: %3.3e, risk %3.3e \n" loss risk
        
    end
    
    return W
end




