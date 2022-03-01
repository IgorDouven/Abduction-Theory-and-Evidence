# enable parallel computing
using Distributed
addprocs(...) # set number of cores you want to make available for the computations

# needed packages
@everywhere using Distributions, Bootstrap, Statistics, LinearAlgebra, SharedArrays
using DataFrames, HypothesisTests, DelimitedFiles, StatsBase, Colors, Gadfly, Cairo, Compose

# default graphics
Gadfly.push_theme(:default)
set_default_plot_size(9inch, 9inch/MathConstants.golden)

function gen_brew_colors(n) # to create your own colors, here based on one of the brewer series
    cs = distinguishable_colors(n,
        [colorant"#66c2a5", colorant"#fc8d62", colorant"#8da0cb", colorant"#e78ac3",
            colorant"#a6d854", colorant"#ffd92f", colorant"#e5c494", colorant"#b3b3b3"],
        lchoices=Float64[58, 45, 72.5, 90],
        transform=c->deuteranopic(c, 0.1),
        cchoices=Float64[20,40],
        hchoices=[75,51,35,120,180,210,270,310]
    )
    convert(Vector{Color}, cs)
end

# set parameters, define priors, etc.
@everywhere begin
    const numb_hyp = 11
    const numb_toss = 500
    const numb_sim = 1000
    const prior = fill(Float32(1/numb_hyp), numb_hyp)
    const likelihood_heads = range(0f0, stop=1, length=numb_hyp)
    const likelihood_tails = range(1f0, stop=0, length=numb_hyp)
end

@everywhere datFunc(bias) = rand(Bernoulli(bias), numb_toss)

# Bayes' rule
@everywhere function b_upd(probs::Array{Float32,1}, dat::Array{Bool,1}, toss_num::Int64)
    if dat[toss_num] == true
        @. (probs * likelihood_heads) / $dot(probs, likelihood_heads)
    else
        @. (probs * likelihood_tails) / $dot(probs, likelihood_tails)
    end
end

# EXPL
@everywhere function expl_upd(probs::Array{Float32,1}, dat::Array{Bool, 1}, toss_num::Int64, bonus::Float32=0.1)
    val::Float32 = mean(dat[1:toss_num]) * 10 + 1
    vec::Array{Float32,1} = if dat[toss_num] == true
            @. (probs * likelihood_heads) / $dot(probs, likelihood_heads)
        else
            @. (probs * likelihood_tails) / $dot(probs, likelihood_tails)
        end

    if val % 1 == .5
        vec[floor(Int64, val)] += .5*bonus
        vec[ceil(Int64, val)] += .5*bonus
    else
        vec[round(Int64, val, RoundNearestTiesAway)] += bonus
    end

    return vec / (1 + bonus)
end

# Good's rule
@everywhere function good_bonus(probs::Array{Float32,1}, res::Bool, λ=2) # with λ=2, we obtain the rule L2 from Douven and Schupbach, Frontiers ...

    pE::Float32 = res == true ? dot(probs, likelihood_heads) : dot(probs, likelihood_tails)
    gb::Array{Float32,1} = res == true ? log.(likelihood_heads ./ pE) : log.(likelihood_tails ./ pE)

    function rsc(i)
        if i >= 0
            1 - exp(2λ^2 * -i^2)
        else
            -1 + exp(2λ^2 * -i^2)
        end
    end

    return map(rsc, gb)

end

@everywhere function good_upd(probs::Array{Float32,1}, dat::Array{Bool,1}, toss_num::Int64, γ::Float32=0.1)

    res::Bool = dat[toss_num]

    probvec::Array{Float32,1} = if res == true
        @. (probs * likelihood_heads) / $dot(probs, likelihood_heads)
    else
        @. (probs * likelihood_tails) / $dot(probs, likelihood_tails)
    end

    goodvec::Array{Float32,1} = probvec + γ .* (probvec .* good_bonus(probs, res))

    return goodvec / sum(goodvec)

end

# Popper's rule
@everywhere function pop_bonus(probs::Array{Float32,1}, res::Bool)

    pE::Float32 = res == true ? dot(probs, likelihood_heads) : dot(probs, likelihood_tails)
    pb::Array{Float32, 1} = res == true ? (likelihood_heads .- pE) ./ (likelihood_heads .+ pE) : (likelihood_tails .- pE) ./ (likelihood_tails .+ pE)

 end

@everywhere function pop_upd(probs::Array{Float32,1}, dat::Array{Bool, 1}, toss_num::Int64, γ::Float32=0.1)

    res::Bool = dat[toss_num]

    probvec::Array{Float32,1} = if res == true
        @. (probs * likelihood_heads) / $dot(probs, likelihood_heads)
    else
        @. (probs * likelihood_tails) / $dot(probs, likelihood_tails)
    end

    popvec::Array{Float32,1} = probvec + γ .* (probvec .* pop_bonus(probs, res))

    return popvec / sum(popvec)

end

# simulations
@everywhere const numb_agents = 200
@everywhere const numb_generations = 250
const numb_simulations = 50

@everywhere function survWei(upds::Array{Float32,2}, # modeling probability of death, based on Weibull distribution
                             hyp::Int64,
                             a::Float64,
                             b::Float64,
                             thresh::Float32,
                             shape::Float64=rand(Uniform(.5, 5)),
                             scale::Float64=rand(Uniform(50, 250))
                             )

    t = something(findfirst(upds .> thresh), (numb_toss, 0)) # where in the matrix with probability updates do we find the first value above thresh?
    c = t[2]
    p = t[1]

    # cdf(Weibull(shape, scale), p) below gives the probability of death at the relevant time

    if c == hyp
        1 - (cdf(Weibull(shape, scale), p) / a) # probability goes down if right intervention is made (which is made when the truth is assigned a probability above thresh)
    elseif c == 0
        1 - cdf(Weibull(shape, scale), numb_toss + 1) # if no intervention is made, output survival probability at last time step
    else
        (1 - cdf(Weibull(shape, scale), p)) / b # probability goes down if wrong intervention is made (which happens if a false hypothesis is assigned a probabilty above thresh)
    end
end

@everywhere function survGam(upds::Array{Float32,2}, # modeling probability of death, based on Gamma distribution
                             hyp::Int64,
                             a::Float64,
                             b::Float64,
                             thresh::Float32,
                             shape::Float64=rand(Uniform(10, 16)),
                             scale::Float64=rand(Uniform(10, 16))
                             )

    t = something(findfirst(upds .> thresh), (numb_toss, 0)) # where in the matrix with probability updates do we find the first value above thresh?
    c = t[2]
    p = t[1]

    if c == hyp
        1 - (cdf(Gamma(shape, scale), p) / a) # the probability goes down if the right intervention is made (and the right intervention is made if the truth is assigned a probability above thresh)
    elseif c == 0
        1 - cdf(Gamma(shape, scale), numb_toss + 1) # if no intervention is made, output survival probability at last time step
    else
        (1 - cdf(Gamma(shape, scale), p)) / b # the probability goes down if the wrong intervention is made (which happens if a false hypothesis is assigned a probabilty above thresh)
    end
end

@everywhere function patient(rule_index::Float32, c_value::Float32, thresh::Float32, dist::Function)

    rand_hyp::Int64 = rand(1:11) # pick α hypothesis ("what's wrong with the patient")
    right = rand(Uniform(1, 10)) # effect of right intervention
    wrong = rand(Uniform(1, 10)) # effect of wrong intervention

    data::Array{Bool, 1} = datFunc((rand_hyp - 1) / (numb_hyp - 1)) # generate synthetic data for this pick (the test results for the patient)

    updates = Array{Float32,2}(undef, numb_toss + 1, numb_hyp) # initialize array for probabilities

    updates[1, :] = prior # set prior

    if rule_index == 1.0f0
        @fastmath @inbounds for t in 1:numb_toss # generate updates
            updates[t + 1, :] = b_upd(updates[t, :], data, t)
        end
    elseif rule_index == 2.0f0
        @fastmath @inbounds for t in 1:numb_toss # generate updates
            updates[t + 1, :] = expl_upd(updates[t, :], data, t, c_value)
        end
    elseif rule_index == 3.0f0
        @fastmath @inbounds for t in 1:numb_toss # generate updates
            updates[t + 1, :] = good_upd(updates[t, :], data, t, c_value)
        end
    else
        @fastmath @inbounds for t in 1:numb_toss # generate updates
            updates[t + 1, :] = pop_upd(updates[t, :], data, t, c_value)
        end
    end

    return dist(updates, rand_hyp, right, wrong, thresh)
end

#= tests doctor on 100 patients and calculates average survival score obtained by doctor, so
average probability that patients would survive =#
@everywhere function avScore(rule_index::Float32, c_value::Float32, thresh::Float32, dist::Function)
    tot = @distributed (+) for i in 1:100
        patient(rule_index, c_value, thresh, dist)
    end
    return tot / 100
end

#= tests all doctors in a population and selects the 50 percent fittest; outputs those
as well as a copy of each of them =#
function population_upd_rep(pop::Array{Float32,2}, dist::Function)
    agent_scores = SharedArray{Float32,1}(numb_agents)
    @sync @distributed for i in 1:numb_agents
        @inbounds agent_scores[i] = avScore(pop[i, :]..., dist)
    end
    best_index = findall(agent_scores .>= Statistics.median(agent_scores))
    best = pop[best_index[1:Int(numb_agents/2)], :]
    return vcat(best, best), agent_scores
end

run(`mkdir data`)

function sim_run(dist::Function)
    k = 1
    while k < numb_simulations + 1
        groupID = repeat(1.0:4.0, inner=div(numb_agents, 4))
        population_start = vcat(fill(0, div(numb_agents, 4)), rand(Uniform(0, .25), 3*div(numb_agents, 4)))
        agent_thresh = rand(Uniform(.5, 1), numb_agents)
        pop_start = Array{Float32,2}(hcat(groupID, population_start, agent_thresh))
        pop_upd_c_a = Array{Float32,3}(undef, numb_agents, 3, numb_generations + 1)
        pop_upd_f = Array{Float32,2}(undef, numb_agents, numb_generations + 1)
        pop_upd_c_a[:, :, 1] = pop_start

        @inbounds for i in 1:numb_generations
            pop_upd_c_a[:, :, i + 1], pop_upd_f[:, i] = population_upd_rep(pop_upd_c_a[:, :, i], dist)
        end

        @inbounds for i in 1:numb_agents
            pop_upd_f[i, numb_generations + 1] = avScore(pop_upd_c_a[i, 1:3, numb_generations + 1]..., dist)
        end

        res_a = Array{Int32,2}(undef, numb_agents, numb_generations + 1)
        res_c = Array{Float32,2}(undef, numb_agents, numb_generations + 1)
        res_t = Array{Float32,2}(undef, numb_agents, numb_generations + 1)

        @inbounds for i in 1:(numb_generations + 1)
            res_a[:, i], res_c[:, i], res_t[:, i] = pop_upd_c_a[:, 1, i], pop_upd_c_a[:, 2, i], pop_upd_c_a[:, 3, i]
        end

        writedlm("data/type$dist$k.txt", res_a)
        writedlm("data/c_value$dist$k.txt", res_c)
        writedlm("data/thresh$dist$k.txt", res_t)
        writedlm("data/fit$dist$k.txt", pop_upd_f)

        population_start = nothing
        pop_start = nothing
        pop_upd_c_a = nothing
        pop_upd_f = nothing
        res_a = nothing
        res_c = nothing
        GC.gc()

        k += 1
    end
end

sim_run(survWei)

fullType = readdlm("data/typesurvWei1.txt")

ks = [ keys(sort(countmap(fullType[:,i]))) for i in 1:numb_generations + 1 ]
vls = [ values(sort(countmap(fullType[:,i]))) for i in 1:numb_generations + 1 ]

group = []
freq = []
gen = []

for i in 1:numb_generations + 1
    append!(group, collect(ks[i]))
    append!(freq, collect(vls[i]))
    append!(gen, fill(i, length(collect(ks[i]))))
end

bar_df = DataFrame(group=group, freq=freq, gen=gen)

bar_df[!, :name] = map(bar_df[!, :group]) do x
    if x == 1
        return "Bayes"
    elseif x == 2
        return "EXPL"
    elseif x == 3
        return "Good"
    else
        return "Popper"
    end
end

rename!(bar_df, [Symbol("$i") for i in ["Group", "Count", "Generation", "Rule"]])

p1 = plot(bar_df, x=:Generation, y=:Count, color=:Rule, Geom.bar(position=:stack),
    Coord.cartesian(xmin=1, xmax=numb_generations + 1),
    Guide.colorkey(title="Rule"),
    Scale.color_discrete_manual(gen_brew_colors(4)...),
    style(minor_label_font_size=10pt, major_label_font_size=14pt,
            key_label_font_size=11pt, key_title_font_size=13pt,
            grid_color=colorant"#222831",
            colorkey_swatch_shape=:square))

p1 |> PDF("exa_with_var_thresh.pdf")

full_results_type = Array{Int32, 3}(undef, numb_agents, numb_generations + 1, numb_simulations)

for i in 1:numb_simulations
    full_results_type[:, :, i] = readdlm("nthresh_data/typesurvWei$i.txt")
end

percentages = Array{Float64, 3}(undef, numb_simulations, numb_generations + 1, 4)

for k in 1:4
    for j in 1:numb_generations + 1
        percentages[:, j, k] = [ length(findall(full_results_type[:, j, i] .== k)) / numb_agents for i in 1:numb_simulations ]
    end
end

@everywhere function bs(x, n=1000)
    bootstrap(mean, x, BasicSampling(n))
end

bayesBoot = SharedArray{Float64}(numb_generations + 1, 3);

@distributed for i in 1:numb_generations + 1
    bayesBoot[i, :] = [Bootstrap.confint(bs(percentages[:, i, 1]), BasicConfInt(.95))[1]...]
end

explBoot = SharedArray{Float64}(numb_generations + 1, 3);

@distributed for i in 1:numb_generations + 1
    explBoot[i, :] = [Bootstrap.confint(bs(percentages[:, i, 2]), BasicConfInt(.95))[1]...]
end

goodBoot = SharedArray{Float64}(numb_generations + 1, 3);

@distributed for i in 1:numb_generations + 1
    goodBoot[i, :] = [Bootstrap.confint(bs(percentages[:, i, 3]), BasicConfInt(.95))[1]...]
end

popperBoot = SharedArray{Float64}(numb_generations + 1, 3);

@distributed for i in 1:numb_generations + 1
    popperBoot[i, :] = [Bootstrap.confint(bs(percentages[:, i, 4]), BasicConfInt(.95))[1]...]
end

Generation = repeat(collect(1:numb_generations + 1), outer=4)
Rule = repeat(["Bayes", "EXPL", "Good", "Popper"], inner=numb_generations + 1)
res = vcat(bayesBoot, explBoot, goodBoot, popperBoot)
type_df = DataFrame(res, :auto)
type_df.Generation=Generation
type_df.Rule=Rule
rename!(type_df, [Symbol("$i") for i in ["y", "ymin", "ymax", "Generation", "Rule"]]);

p2 = plot(type_df, x=:Generation, y=:y, ymin=:ymin, ymax=:ymax, color=:Rule, Geom.line, Geom.ribbon,
    Guide.ylabel("Average percentage"),
    Coord.cartesian(xmin=1, xmax=numb_generations + 1, ymin=-.001),
    Scale.color_discrete_manual(gen_brew_colors(4)...),
    style(line_width=2pt, lowlight_color=c->RGB{Float32}(c.r, c.g, c.b), alphas=[.2,.2,.2,.2],
          minor_label_font_size=10pt, major_label_font_size=14pt,
          key_label_font_size=11pt, key_title_font_size=13pt))

p2 |> PDF("perc_with_var_thresh.pdf")

p = plot(type_df, x=:Generation, y=:y, color=:Rule, Geom.line,
    Guide.ylabel("Average percentage"),
    Coord.cartesian(xmin=1, xmax=numb_generations + 1, ymin=-.001),
    Scale.color_discrete_manual(gen_brew_colors(4)...),
    style(line_width=2pt,
            minor_label_font_size=10pt, major_label_font_size=14pt,
            key_label_font_size=11pt, key_title_font_size=13pt,
            colorkey_swatch_shape=:square))
        
p |> PDF("percentages_thresh.pdf")

full_results_t = Array{Float64, 3}(undef, numb_agents, numb_generations + 1, numb_simulations)

for i in 1:numb_simulations
    full_results_t[:, :, i] = readdlm("nthresh_data/threshsurvWei$i.txt")
end

thresh_res = Array{Float64, 2}(undef, numb_generations + 1, 3)

for k in 1:numb_generations + 1
    cr = Float64[]
    for j in 1:numb_simulations
        push!(cr, mean([ full_results_t[:, k, i] for i in 1:numb_simulations ][j]))
    end
    cr = cr[.!isnan.(cr)]
    thresh_res[k, :] = vcat(mean(cr), [confint(OneSampleTTest(cr))...])
end

mns = mean_and_std.([ full_results_t[:, i, :][:] for i in 1:numb_generations + 1 ])

thresh_df = DataFrame(mean=first.(mns), std=last.(mns))
thresh_df.gen = 1:size(thresh_df, 1)
thresh_df.min = thresh_df.mean .- thresh_df.std
thresh_df.max= thresh_df.mean .+ thresh_df.std

q = plot(thresh_df, x=:gen, y=:mean, ymin=:min, ymax=:max, Geom.line, Geom.ribbon,
         #Coord.cartesian(xmin=1, xmax=25),
         Scale.x_log2,
         Guide.xlabel("Generation"), Guide.ylabel("Average threshold value"),
         Theme(line_width=2pt, default_color = RGB(102/255,194/255,165/255),
               minor_label_font_size=10pt, major_label_font_size=14pt, alphas=[.2],
               key_label_font_size=11pt, key_title_font_size=13pt))

q |> PDF("thresholds.pdf")

# thresholds per agent type
bayes_thresh = [ mean_and_std(vcat([ full_results_t[:, j, i][full_results_type[:, j, i] .== 1] for i in 1:numb_simulations ]...)) for j in 1:numb_generations + 1 ]
expl_thresh = [ mean_and_std(vcat([ full_results_t[:, j, i][full_results_type[:, j, i] .== 2] for i in 1:numb_simulations ]...)) for j in 1:numb_generations + 1 ]
good_thresh = [ mean_and_std(vcat([ full_results_t[:, j, i][full_results_type[:, j, i] .== 3] for i in 1:numb_simulations ]...)) for j in 1:numb_generations + 1 ]
popper_thresh = [ mean_and_std(vcat([ full_results_t[:, j, i][full_results_type[:, j, i] .== 4] for i in 1:numb_simulations ]...)) for j in 1:numb_generations + 1 ]

Generation = repeat(collect(1:numb_generations + 1), outer=4)
Rule = repeat(["Bayes", "EXPL", "Good", "Popper"], inner=numb_generations + 1)
t_res = vcat(bayes_thresh, expl_thresh, good_thresh, popper_thresh)
t_df = DataFrame(y = first.(t_res), ymin = first.(t_res) .- last.(t_res), ymax = first.(t_res) .+ last.(t_res), Generation = Generation, Rule = Rule)

p_thresh = plot(t_df, x=:Generation, y=:y, ymin=:ymin, ymax=:ymax, color=:Rule, Geom.line, Geom.ribbon,
    Guide.ylabel("Average threshold value"),
    #Coord.cartesian(xmin=1, xmax=numb_generations + 1, ymin=-.001),
    Scale.x_log2,
    Scale.color_discrete_manual(gen_brew_colors(4)...),
    style(line_width=2pt, lowlight_color=c->RGB{Float32}(c.r, c.g, c.b), alphas=[.2,.2,.2,.2],
          minor_label_font_size=10pt, major_label_font_size=14pt,
          key_label_font_size=11pt, key_title_font_size=13pt))

p_thresh |> PDF("thresh_per_type.pdf")
