## Problem: Job sequencing and tool switching problem (SSP) - Catanzaro, Gouviea, Labbe (2015) formulation
## Solver: Gurobi
## Language: Julia (JuMP)
## Written by: @setyotw
## Date: Sept 2, 2022

#%% import packages
using Pkg, JuMP, Gurobi, DataStructures, Combinatorics
Pkg.status()
import FromFile: @from

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  DEVELOPMENT PARTS
function UniformSSP_Catanzaro_Formulation(instanceSSP, magazineCap, MILP_Limit)
    # 1 | initialize sets and notations
    # number of available jobs (horizontal)
    n = length(instanceSSP[1,:])
    # number of available tools (vertical)
    m = length(instanceSSP[:,1])

    J = [i for i in range(1, n)] ## list of jobs
    J0 = [i for i in range(0, n)] ## list of positions
    T = [i for i in range(1, m)] ## list of tools
    T0 = [i for i in range(0, m)] ## list of tools + dummy zeros
    V = [i for i in range(1,C)] ## list of magazine slots
    
    arcIJ = [(i,j) for i in J0 for j in J0]
    arcIJT = [(i,j,t) for i in J0 for j in J0 if i!=j for t in T]
    
    Tj = Dict((j) => [] for j in J0)
    TnotTj = Dict((j) => [] for j in J0)
    for job in J
        for tools in T
            if instanceSSP[tools,job] == 1
                append!(Tj[job], tools)
            else
                append!(TnotTj[job], tools)
            end
        end
    end

    # 2 | initialize parameters
    C = Int(magazineCap)
    
    # 3 | initialize the model
    model = Model(Gurobi.Optimizer)

    # 4 | initialize decision variables
    @variable(model, X[arcIJ], Bin) # X[ij] = Equal to 1 if job j processed in position k
    @variable(model, Y[arcIJT], Bin) # Y[ijt] = Equal to 1 if tool t presents in the transition between jobs i and j
    @variable(model, Z[arcIJT], Bin) # Z[ijt] = tool switch, equal to 1 if tool t has to be added in the transition between jobs i and j
    
    # 5 | define objective function
    @objective(model, Min, 
        sum(Z[(i,j,t)] for i in J0 for j in J0 if i!=j for t in Tj[j]))

    # 6 | define constraints
    for i in J0
        @constraint(model, sum(X[(i,j)] for j in J0 if j!=i) == 1)
    end

    for j in J0
        @constraint(model, sum(X[(i,j)] for i in J0 if i!=j) == 1)
    end
 
    subtourSet = collect(powerset(J0))[2:end-1]
    for S in subtourSet
        @constraint(model, sum(X[(i,j)] for i in S for j in S if i!=j) <= length(S)-1)
    end
    
    for j in J0
        for t in Tj[j]
            @constraint(model, sum((Y[(i,j,t)]+Z[(i,j,t)]) for i in J0 if i!=j) == 1)
        end
    end

    for i in J0
        for j in J0
            for t in Tj[j]
                if i!=j && t ∉ Tj[i]
                    @constraint(model, Y[(i,j,t)] + Z[(i,j,t)] <= X[(i,j)])
                end
            end
        end
    end
    
    for i in J0
        for j in J0
            if i!=j
                @constraint(model, sum(Y[(i,j,t)]+Z[(i,j,t)] for t in T) <= C*X[(i,j)])
            end
        end
    end

    for i in J0
        for t in T
            @constraint(model, sum(Y[(k,i,t)]+Z[(k,i,t)] for k in J0 if k!=i) >= sum(Y[(i,j,t)] for j in J0 if j!=i))
        end
    end

    for i in J0 
        for j in J0 
            for t in Tj[i] 
                if i!=j && t in Tj[j]
                    @constraint(model, Y[(i,j,t)] >= X[(i,j)])
                end
            end
        end
    end
    
    for i in J0
        for j in J0
            for t in TnotTj[j]
                if i!=j
                    @constraint(model, Z[(i,j,t)] == 0)
                end
            end
        end
    end

    for j in J
        for t in T
            @constraint(model, Y[(0,j,t)] == 0)
        end
    end

    # 7 | call the solver (we use Gurobi here, but you can use other solvers i.e. PuLP or CPLEX)
    JuMP.set_time_limit_sec(model, MILP_Limit)
    JuMP.optimize!(model)

    # 8 | extract the results    
    completeResults = solution_summary(model)
    solutionObjective = objective_value(model)
    solutionGap = relative_gap(model)
    runtimeCount = solve_time(model)
    all_var_list = all_variables(model)
    all_var_value = value.(all_variables(model))
    X_active = [string(all_var_list[i]) for i in range(1,length(all_var_list)) if all_var_value[i] > 0 && string(all_var_list[i])[1] == 'X']
    Y_active = [string(all_var_list[i]) for i in range(1,length(all_var_list)) if all_var_value[i] > 0 && string(all_var_list[i])[1] == 'Y']
    Z_active = [string(all_var_list[i]) for i in range(1,length(all_var_list)) if all_var_value[i] > 0 && string(all_var_list[i])[1] == 'Z']
    
    return solutionObjective, solutionGap, X_active, Y_active, Z_active, runtimeCount, completeResults
end

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  IMPLEMENTATION PARTS
#%% input problem instance
# a simple uniform SSP case with 5 different jobs, 6 different tools, and 3 capacity of magazine (at max, only 3 different tools could be installed at the same time)
instanceSSP = Array{Int}([
        1 1 0 0 1;
        1 0 0 1 0;
        0 1 1 1 0;
        1 0 1 0 1;
        0 0 1 1 0;
        0 0 0 0 1])

initialSetupSSP = Array{Int}([2 1 1 2 3 2 3 3 2 3 1 3 3 2 2 2 3 3 1 1])

magazineCap = Int(3)

#%% termination time for the solver (Gurobi)
MILP_Limit = Int(3600)

#%% implement the mathematical formulation
# solutionObjective --> best objective value found by the solver
# solutionGap --> solution gap, (UB-LB)/UB
# U_active, V_active, W_active --> return the active variables
# runtimeCount --> return the runtime in seconds
# completeResults --> return the complete results storage
solutionObjective, solutionGap, X_active, Y_active, Z_active, runtimeCount, completeResults = UniformSSP_Catanzaro_Formulation(instanceSSP, magazineCap, MILP_Limit)