
using JuMP
using Ipopt
using CalculusWithJulia
using DelimitedFiles

# GIVEN: a minimization problem: min|y(T ) - y_ d|_(L^2) ^2 + |u|_(L^2) ^2 governed by an
# inhomogeneous heat equation y_t - laplace y = -u*Chi, Chi is an indicator function, 
# u(t) in [u_a, u_b] (Box constraints).

# GOAL: solve this problem with semismooth newton method. 
# --> Find the roots of the optimality system (KKT system) f(x) = 0 by semi smooth newton method.
# A newton step is: x_new = x_old - grad(f(x_old)^-1 * f(x_old)
# We need a function f(x) that is the optimality system as a function of x = (y,p,u), where p is the adjoint
# and we need the gradient of f(x). Since f(x) is not sufficiently smooth due to the optimality conditions 
# coming from the box constraints on u, we need to consider the subgradient instead of the gradient.



function y_heat_ipopt(y_0, u, omega1, omega2, N, T)
    y_heat = Model(optimizer_with_attributes(Ipopt.Optimizer))#, "print_level"=>12))
    #set_optimizer_attribute(y_heat, "constr_viol_tol", 1e-6)  # Adjust the value as needed
    set_optimizer_attribute(y_heat, "warm_start_init_point", "yes")

    h = 1.0/N
    tau = 1.0/T
    Chi = [((omega1 <= i * h <= omega2) ? 1 : 0) for i in 0:N]

    @variable(y_heat, Y[1:N+1,1:T+1])            #+1 due to the 0
    
    #Dirichlet boundary
    @constraint(y_heat, [j = 1:T+1], Y[1, j]==0.0)
    @constraint(y_heat, [j = 1:T+1], Y[N+1, j]==0.0)
    #initial condition
    @constraint(y_heat, [i=2:N], Y[i,1] == y_0[i]) #Für Eigenlösung
    #PDE
    for j in 1:T, i in 2:N
        # implizit Diskretisierung
        @constraint(y_heat, Y[i,j+1] == Y[i,j] + tau/h^2*(Y[i-1,j+1] - 2*Y[i,j+1] + Y[i+1,j+1])
                            - tau*Chi[i]*u[j])
    end

    JuMP.optimize!(y_heat)
    return value.(Y)
end



function p_solver(Y, y_d, N, T)
    model_p = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0))
    set_optimizer_attribute(model_p, "warm_start_init_point", "yes")

    h = 1.0/N
    tau = 1.0/T

    @variable(model_p, P[1:N+1,1:T+1])            #+1 due to the 0

    #Dirichlet boundary
    @constraint(model_p, [j = 1:T+1], P[1, j]==0.0)
    @constraint(model_p, [j = 1:T+1], P[N+1, j]==0.0)

    #end condition
    @constraint(model_p, [i = 2:N], P[i,T+1] == -(Y[i, T+1]- y_d[i]))

    #PDE
    for j in 1:T, i in 2:N
        # implizites Euler
        @constraint(model_p, P[i,j+1] == P[i, j] - (tau/h^2)*(P[i+1,j] - 2*P[i,j] + P[i-1,j]) )
    end

    optimize!(model_p)
    return value.(P)
end


function nonlin_prblm_ipopt( y_0, y_d, omega1, omega2, alpha, N, T)
    
    model = Model(optimizer_with_attributes(Ipopt.Optimizer))#, "print_level"=>12)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-6)  # Adjust the value as needed
    set_optimizer_attribute(model, "warm_start_init_point", "yes")

    h = 1.0/N
    tau = 1.0/T
    Chi = [((omega1 <= i*h <= omega2) ? 1 : 0) for i in 0:N]

    # first component indicates the space, the second component indicates the time
    @variable(model, Y[1:N+1,1:T+1])            #+1 due to the 0
    @variable(model, u[1:T])

    #Dirichlet boundary
    @constraint(model, [j = 1:T+1], Y[1, j]==0.0)
    @constraint(model, [j = 1:T+1], Y[N+1, j]==0.0)

    #initial condition
    @constraint(model, [i=2:N], Y[i,1] == y_0[i])

    # bounds on u and v 
    @constraint(model, [j = 1:T], -100 <= u[j] <= 100)
    
    #PDE
    for j in 1:T, i in 2:N
        # implizite Diskretisierung
        @constraint(model, Y[i,j+1] == Y[i,j] + tau/h^2*(Y[i-1,j+1] - 2*Y[i,j+1] + Y[i+1,j+1])
                            - tau*Chi[i]*u[j])
    end

    #nonlinear problem
    @NLobjective(model, Min,  h/2*(sum((Y[i, T+1] - y_d[i])^2 for i in 2:N)) 
            + h/4*(Y[1, T+1] - y_d[1])^2 + h/4*(Y[N+1, T+1] - y_d[N+1])^2 
            + (1/2)*alpha*tau*(sum(u[j]^2 for j in 2:T-1)) + (alpha/2*tau/2)*u[1]^2 + (alpha/2*tau/2)*u[T]^2)

#
    JuMP.optimize!(model)
    return value.(Y), value.(u), objective_value(model)
end


function Opti_System(y, p ,u , y_d, omega1, omega2, u_a, u_b, alpha, N, T)
    # Optimality system of the optimization problem, discretized with implicit finite differences,
    # written as a function (y, p, u) -> S(y,p,u)
    
    kkt = zeros((2*(N+1)*(T+1)+T))

    tau = 1.0/T             # Gitterweite Zeit
    h = 1.0/N               # Gitterweite Ort

    # define the indicator function Chi
    Chi = [((omega1 <= i*h <= omega2) ? 1 : 0) for i in 0:N]
    
    # initial contition
    y_0 = zeros(N+1)

    # Randbedingung
    for j in 1:T+1
	    kkt[((j-1)*(N + 1) +1)] = y[1,j] # x[(j-1)*(N +1) +1]          #y[0,j] = 0
	    kkt[j*(N + 1)] = y[N+1,j] # x[j*(N + 1)]                       #y[N,j] = 0

        kkt[((j+(T+1)-1)*(N + 1) +1)] = p[1,j] #x[(j-1)*(N +1) +1]          #p[0,j] = 0
        kkt[(j+(T+1))*(N + 1)] = p[N+1,j] # x[j*(N + 1)]                       #p[N,j] = 0
    end


    # Anfangsbedingungen (Vorwärtsgleichung) y[i,0]=y_0
    for i in 2:N	# An den Stellen 1 und N+1 greift die Randbedingung
        kkt[i] = y[i,1] - y_0[i]	
    end

    # Vorwärtsgleichung (PDE)
    for j in 1:T
        for i in 2:N   
            kkt[i+(j-1)*(N+1)] =( y[i,j+1] - y[i,j] -(tau/h^2)*(y[i-1,j+1] -2*y[i,j+1] + y[i+1,j+1])
                                + tau*Chi[i]*u[j] )
        end
    end
    


    # Endbedingung,  Adjungierte Gleichung
    for i in 2:N
        kkt[i + (N+1)*(T+1)+(N+1)*T] = p[i, T+1] + (y[i,T+1] -y_d[i])	# An den Stellen 1 und N+1 greift die Randbedingung	

    end

    # Adjungierte Gleichung P[i,j+1] == P[i, j] - (tau/h^2)*(P[i+1,j] - 2*P[i,j] + P[i-1,j]) 
    for j in 1:T
        for i in 2:N
            kkt[i+(j-1 + T+1)*(N+1)] = ((p[i, j+1] - p[i,j]) + (tau/h^2)*(p[i+1, j] - 2*p[i,j] + p[i-1, j])  )  
        end	
    end

    # Box constraints on u 
    for j in 1:T
        #kkt[j+2*(T+1)*(N+1)] = alpha*u[j] + h*sum(Chi[i]*p[i,j] for i in 1:N+1)

        kkt[j+2*(T+1)*(N+1)] = alpha*u[j] + min(alpha*u_b[j], max(alpha*u_a[j], h*sum(Chi[i]*p[i,j] for i in 1:N+1)))
    end

    writedlm("opti-system.txt", kkt)

    return kkt

end


function Jcb_Opti_System(omega1, omega2,p, u_a, u_b, alpha, N, T) #The dependency on p is due to the subgradient
    # Jacobi matrix of the optimality system, by hand to be sure what exactely is happening in the background.

    tau = 1.0/T             # Gitterweite Zeit
    h = 1.0/N               # Gitterweite Ort
    # define the indicator function Chi
    Chi = [((omega1 <= i*h <= omega2) ? 1 : 0) for i in 0:N]

    # initial contition
    #y_0 = zeros(N+1)
    
    # initialize Jacobi-Matrix as 0-Matrix, we fill the Matrix with the right values
    jac = fill(0.0, (N+1)*(T+1) + (N+1)*(T+1) + T, (N+1)*(T+1) + (N+1)*(T+1) +T)

    # build the subgradient of the projection term
    g = zeros(T)
    for j in 1:T
        x = h*sum(Chi[i]*p[i,j+1] for i in 1:N+1)
        if isless(x, alpha*u_a[j]) || isless(alpha*u_b[j], x)
            g[j] = 0
        
        elseif isless(alpha*u_a[j], x) || isless(x, alpha*u_b[j])
            g[j] = 1 
        
        else
            g[j] = 0.5
        end
    end


    ## Anfangsbedingung
    for i in 2:N
        # Ableitung nach y  = 1
        jac[i, i] = 1

        # Ableitung nach p = 0
        # Ableitung nach u = 0
    end
   

    ## Vorwärtsgleichung: y[i,j+1] - y[i,j] -(tau/h^2)*(y[i-1,j+1] -2*y[i,j+1] + y[i+1,j+1])+ tau*Chi[i]*u[j]
    ## Randbedingungen
    for j in 0: T
        # Ableitung nach y
        jac[1 + j*(N+1), 1 + j*(N+1)] = 1
        jac[N+1 + j*(N+1), N+1 + j*(N+1)] = 1
    end
    
    for j in 1:T
        for i in 2:N
            # Ableitung nach y

            jac[i + (j)*(N+1), i + (j)*(N+1)] = tau/h^2*2 + 1
            jac[i + (j)*(N+1), i-1 + (j)*(N+1)] = -tau/h^2
            jac[i + (j)*(N+1), i+1 + (j)*(N+1)] = -tau/h^2
            jac[i + (j)*(N+1), i + (j-1)*(N+1)] = -1

            # Ableitung nach  p = 0

            # Ableitung nach u
            jac[i+j*(N+1), 2*(N+1)*(T+1)+j] = tau*Chi[i]
        end
    end

    ## Rückwärtsgleichung: (p[i, j+1] - p[i,j]) + (tau/h^2)*(p[i+1, j] - 2*p[i,j] + p[i-1, j])
    #Randbedingungen
    for j in 0:T
        # Ableitung nach p
        jac[(1 + (j+T+1)*(N+1)), (1 + (j+T+1)*(N+1))] = 1
        jac[(N+1 + (j+T+1)*(N+1)), (N+1 + (j+T+1)*(N+1))] = 1    
    end

    for j in 1:T
        for i in 2:N
            # Ableitung nach y = 0

            # Ableitung nach p
            jac[(i + (N+1)*(T+1)+ (j-1)*(N+1)), (i + (N+1)*(T+1)+ (j-1)*(N+1))] = -2*tau/h^2 - 1
            jac[(i + (N+1)*(T+1)+ (j-1)*(N+1)),( i-1 + (N+1)*(T+1)+ (j-1)*(N+1))] = 1*tau/h^2
            jac[(i + (N+1)*(T+1)+ (j-1)*(N+1)), (i+1 + (N+1)*(T+1)+ (j-1)*(N+1))] = 1*tau/h^2
            jac[(i + (N+1)*(T+1)+ (j-1)*(N+1)), (i + (N+1)*(T+1)+ (j)*(N+1))] = 1

            # Ableitung nach u = 0
        end
    end

    ## Endbedingung kkt[i + (N+1)*(T+1)+(N+1)*T] = p[i, T+1] + (y[i,T+1] -y_d[i])
    for i in 2:N
        # Ableitung nach y
        jac[(N+1)*(T+1)+(N+1)*T+i, (N+1)*(T)+ i] = 1
        # Ableitung nach p
        jac[(N+1)*(T+1)+(N+1)*T+i, ((N+1)*(T+1) + (N+1)*T +i)] = 1
        # Ableitung nach u = 0
    end
    

    ## design Gleichung: alpha*u[j] + h*sum(Chi[i]*p[i,j] for i in 1:N+1)
    for j in 1:T
        for i in 2:N
            # Ableitung nach y = 0

            # Ableitung nach p 
            jac[2*(N+1)*(T+1)+j, (N+1)*(T+1)+i + (j-1)*(N+1) ] = g[j]*h*Chi[i]
            #jac[2*(N+1)*(T+1)+j, (N+1)*(T+1)+i + j*(N+1) ] = g[j]*(-1/alpha*h*Chi[i])

            # Ableitung nach u
            jac[2*(N+1)*(T+1)+j, 2*(N+1)*(T+1)+j] =  alpha


        end
    end

    writedlm("Jcb-opti-system.txt", jac)

    return jac
    
end


function newton(y_k, p_k, u_k, y_d, omega1, omega2, u_a, u_b, alpha, N, T)
    # reduce the functions such that we don't have to write down every time all the arguments
    reduced_Opti_Sys(y_k, p_k, u_k) = Opti_System(y_k, p_k, u_k, y_d, 
                                                omega1, omega2, u_a, u_b, alpha, N, T)
    reduced_Jcb_Opti_Sys(p_k) = Jcb_Opti_System(omega1, omega2, p_k,
                                                u_a, u_b, alpha, N, T)

    ### First we create an initial x_new and respective Optisystem

    # Left division operator x\y: multiplication of y by the inverse of x on the left. This yields a
    # 2*(N+1)*(T+1)+(T+1) x 1 vector 
    x_new = [reshape(y_k,(N+1)*(T+1),1)
              reshape(p_k,(N+1)*(T+1),1)
              u_k[1:T]]  - 1/2*reduced_Jcb_Opti_Sys(p_k)\ reduced_Opti_Sys(y_k, p_k, u_k)
    
    quotient = 1/2*reduced_Jcb_Opti_Sys(p_k)\ reduced_Opti_Sys(y_k, p_k, u_k)          
    #println("kkt/Jac = $quotient")
    jac = reduced_Jcb_Opti_Sys(p_k)
    inverse = inv(jac)
    #println("inverse of Jacobian: $inverse")

    #extract y_new, p_new and u_new from the x_new vector
    y_new = reshape(x_new[1 : (N+1)*(T+1)], N+1, T+1)
    p_new = reshape(x_new[(N+1)*(T+1)+1: 2*(N+1)*(T+1)],  N+1, T+1)
    u_new = x_new[(2*(N+1)*(T+1)+1):(2*(N+1)*(T+1)+T)]

    Opti_System_new = reduced_Opti_Sys(y_new, p_new, u_new)
    norm_Optisysnew = norm(Opti_System_new)

    println("norm von Opti_System_new: $norm_Optisysnew")


    #now the actual algorithm is starting
    k = 0
    tol = 10^(-3)              

    while (k < 100) && (norm(Opti_System_new) > tol )
        # x_new from last loop is now the old iterate x_old
        x_old = x_new
        y_old = reshape(x_old[1 : (N+1)*(T+1)], N+1, T+1)
        p_old = reshape(x_old[(N+1)*(T+1)+1: 2*(N+1)*(T+1)],  N+1, T+1)
        u_old = x_old[(2*(N+1)*(T+1)+1):(2*(N+1)*(T+1)+T)]

        # Obtain new iterate by solving
        s = - reduced_Jcb_Opti_Sys(p_old) \ reduced_Opti_Sys(y_old, p_old, u_old)
        x_new = x_old + s

        # extract y, p, u from x_new
        y_new = reshape(x_new[1 : (N+1)*(T+1)], N+1, T+1)
        p_new = reshape(x_new[(N+1)*(T+1)+1: 2*(N+1)*(T+1)],  N+1, T+1)
        u_new = x_new[(2*(N+1)*(T+1)+1):(2*(N+1)*(T+1)+T)]

        Opti_System_new = reduced_Opti_Sys(y_new, p_new, u_new)
        norm_Optisysnew = norm(Opti_System_new)


        k = k+1
        println(k)
        println("norm von Opti_System_new: $norm_Optisysnew")

    end
    #println(Opti_System_new)
    #norm_Optisysnew = norm(Opti_System_new)



end


### Probe der Optimalitätsbedingungen: 
#konstruiere ein y_d mit Hilfe von Ipopt.
#Löse das Minimierungsproblem mit Hilfe von Ipopt und erhalte daraus y und u.
#Berechne mithilfe dieses y und y_d das zugehörige p mittels Ipopt.


#Setting 1
Alpha = 10^(-5)

N = 20
T = 20
tau = 1.0/T
h = 1.0/N

y_0 = fill(0.0, T+1) #zeros(N+1) 
om1 = 0. # 0.1
om2 = 1 # 0.6

##### Setting 1

# konstruiere ein y_d
u_d = fill(1, T)
y_d = y_heat_ipopt(y_0, u_d, om1, om2, N, T)[:, T+1]
y_d_full = y_heat_ipopt(y_0, u_d, om1, om2, N, T)

# y_d = zeros(N+1)
# for i in 0:N
#     y_d[i+1] = sum(4/((2*n-1)^3*pi)*(1-exp(-(2*n-1)^2))*sin((2*n-1)*pi*i*h) for n in 1:1000)
# end

writedlm("y_d.txt", y_d)

# bounds on u
u_a = fill(-100, T)
u_b = fill(100, T)

#Berechne ein optimum y, u 
#y_ipopt = y_heat_ipopt(y_0, u_d, om1, om2, N, T)
y_opt, u_opt, obj_val = nonlin_prblm_ipopt( y_0, y_d, om1, om2, Alpha, N, T)

println("u_opt: $u_opt")

writedlm("y_opt.txt", y_opt)
writedlm("u_opt.txt", u_opt)

# Berechne die passende Adjungierte P
p_opt = p_solver(y_opt, y_d, N, T)
#p_d = p_solver(y_d_full, y_d, N,T)
writedlm("p_opt.txt", p_opt)


test_kkt = Opti_System(y_opt, p_opt, u_opt, y_d, om1, om2,u_a, u_b, Alpha, N, T)
norm_kkt = norm(test_kkt)

println("norm von kkt: $norm_kkt")

test_jacobian = Jcb_Opti_System(om1, om2, p_opt, u_a, u_b, Alpha, N, T)
test_newton = newton(y_opt, p_opt, u_opt, y_d, om1, om2, u_a, u_b, Alpha, N, T)

primal = test_kkt[1:(N+1)*(T+1)]
dual = test_kkt[((N+1)*(T+1)+1 : 2*(N+1)*(T+1))]
design = test_kkt[(2*(N+1)*(T+1)+1 : 2*(N+1)*(T+1)+T)]


#println(test_kkt)
norm_primal = norm(primal)
norm_dual = norm(dual)
norm_design = norm(design)

println("norm von primal: $norm_primal")
println("   ")
println("norm von dual: $norm_dual")
println("   ")
println("design: $design")
println("   ")
println("norm von design: $norm_design")









