
#import Pkg; Pkg.add("Plots")
#import Pkg; Pkg.add("DataFrames")
#import Pkg; Pkg.add("CSV")

import Random

using Printf
using Statistics
using CSV, DataFrames
using DelimitedFiles

const ext = true
const parset = "figure1"
const seed = parse(Int64,ARGS[1])

if ext
    const mfpt_name = string("mfpt",parset)

else
    const mfpt_name = string("mfpt",parset,"_noext")
end

# note that switch_v values are from the mean-field model
# so forces and the switch_v must be scaled by 2


println("TEST0")

if mfpt_name == "mfpt1a"
	const p1=4
	const gamma=0.322
	const zeta = .2
	const nX = 100 # 12-100  on each side
	const nY = 100 # 12-100  on each side

	const alpha = 5 # rate constants 1/s
	const beta = 200

	const A = 5 # nm displacement of newly attached crossbridge
	const B = 5.05

	const T0 = 0#10/(alpha+beta) # start time
	const Tf = 50 # final time
	const V0 = 0 # nm/s shortening velocity for t > Tstart

	const switch_v = 61

elseif mfpt_name == "mfpt1b"
	const p1=4/5
	const gamma=0.322
	const zeta = .2
	const nX = 500 # 12-100  on each side
	const nY = 500 # 12-100  on each side

	const alpha = 5 # rate constants 1/s
	const beta = 200

	const A = 5 # nm displacement of newly attached crossbridge
	const B = 5.05

	const T0 = 0#10/(alpha+beta) # start time
	const Tf = 50 # final time
	const V0 = 0 # nm/s shortening velocity for t > Tstart

	const switch_v = 61


elseif mfpt_name == "mfpt1a_noext"

	const p1=4
	const gamma=0.322
	const zeta = .4
	const nX = 100 # 12-100  on each side
	const nY = 100 # 12-100  on each side

	const alpha = 10 # rate constants 1/s
	const beta = 150

	const A = 5 # nm displacement of newly attached crossbridge
	const B = 5.001

	const T0 = 0#10/(alpha+beta) # start time
	const Tf = 50 # final time
	const V0 = 0 # nm/s shortening velocity for t > Tstart
	
	const switch_v = 100


elseif mfpt_name == "mfpt1b_noext"

	const p1=4
	const gamma=0.322
	const zeta = .4
	const nX = 100 # 12-100  on each side
	const nY = 100 # 12-100  on each side

	const alpha = 10 # rate constants 1/s
	const beta = 150

	const A = 5 # nm displacement of newly attached crossbridge
	const B = 5.001

	const T0 = 0#10/(alpha+beta) # start time
	const Tf = 50 # final time
	const V0 = 0 # nm/s shortening velocity for t > Tstart
	
	const switch_v = 100
	
elseif mfpt_name == "mfpt2a_noext"

	const p1=4
	const gamma=0.322
	const zeta = .7
	const nX = 100 # 12-100  on each side
	const nY = 100 # 12-100  on each side

	const alpha = 14 # rate constants 1/s
	const beta = 300

	const A = 5 # nm displacement of newly attached crossbridge
	const B = 5.01

	const T0 = 0#10/(alpha+beta) # start time
	const Tf = 50 # final time
	const V0 = 0 # nm/s shortening velocity for t > Tstart
	
	const switch_v =  37
	
elseif mfpt_name == "mfpt2a"

	const p1=4
	const gamma=0.322
	const zeta = .7
	const nX = 100 # 12-100  on each side
	const nY = 100 # 12-100  on each side

	const alpha = 14 # rate constants 1/s
	const beta = 300

	const A = 5 # nm displacement of newly attached crossbridge
	const B = 5.01

	const T0 = 0#10/(alpha+beta) # start time
	const Tf = 50 # final time
	const V0 = 0 # nm/s shortening velocity for t > Tstart
	
	const switch_v =  36

elseif mfpt_name == "mfptfigure1"

	const p1=4
	const gamma=0.322
	const zeta = .4
	const nX = 100 # 12-100  on each side
	const nY = 100 # 12-100  on each side

	const alpha = 14 # rate constants 1/s
	const beta = 126

	const A = 5 # nm displacement of newly attached crossbridge
	const B = 5.05

	const T0 = 0#10/(alpha+beta) # start time
	const Tf = 25 # final time
	const V0 = 0 # nm/s shortening velocity for t > Tstart
	
	const switch_v =  121

end

	
println("p1=",p1," gamma=",gamma," zeta=",zeta," nX=",nX," nY=",nY," alpha=",alpha," beta=",beta," A=",A," B=",B," T0=",T0," Tf=",Tf," V0=",V0," switch_v=",switch_v," mfpt_name=",mfpt_name)

track_positions = false
attach_time = false
use_storage = false
use_last = false


if attach_time
    track_positions = true
end

function run_agents(dt::Float64)

    Random.seed!(seed)

    println("Initializing with seed=",seed)

    dt::Float64 = dt # time step

    a1 = zeros(UInt8,nX)#{Int64}(undef,nX) # binary variable for motors
    x1 = zeros(Float64,nX)#Array{Float64}(undef,nX) # position for each motor

    a2 = zeros(UInt8,nY)#Array{Int64}(undef,nY) # binary variable for motors
    x2 = zeros(Float64,nY)#Array{Float64}(undef,nY) # position for each motor

    #println(a1[change1])

    U1::Float64 = 0. # total attached
    U2::Float64 = 0. # total attached

    F1::Float64 = 0. # force
    F2::Float64 = 0. # force

    # position and velocity
    #V = V0

    #V = zeros(Float64,convert(Int64,floor(Tf/dt)+3))
    V::Float64 = V0

    switch_times = zeros(Float64,0)
    #println(switch_times)

    side::UInt8 = 0
    k::UInt64 = 1
    t::Float64 = 0.

    randsize::UInt16 = 50

    j::UInt64 = 1
    #ran1 = rand(randsize,nX)
    #ran2 = rand(randsize,nY)

	betadt::Float64 = beta*dt
	alphadt::Float64 = alpha*dt
	p1g::Float64 = p1*gamma

    while t < Tf

        t = (k-1)*dt

        #if !ext
        #    dx1 = 0
        #    dx2 = 0
        #else
        #    dx1 = V*dt
        #    dx2 = -V*dt
        #end
		
		if V >= 0
			dx1 = V*dt
			dx2 = -ext*V*dt
		else
			dx1 = ext*V*dt
			dx2 = -V*dt
		end

        for i = 1:length(a1)
            if a1[i]>0

                x1[i] = x1[i] + dx1

               if ((rand() < betadt) | (x1[i] > B))
               #if ((ran1[j,i] < (beta*dt)) | (x1[i] > B))
                    a1[i] = 0
                    x1[i] = 0
                end
            else
                if (rand() < alphadt)# & (V < 0)
                #if (ran1[j,i] < (alpha*dt))
                    a1[i] = 1
                    x1[i] = A
                end
            end
        end

        for i = 1:length(a2)
            if a2[i] > 0

                x2[i] = x2[i] + dx2

                if ((rand() < betadt) | (x2[i] > B))
                #if ((ran2[j,i] < (beta*dt)) | (x2[i] > B))
                    a2[i] = 0
                    x2[i] = 0
                end
            else
                if (rand() < alphadt)# & (V > 0)
                #if (ran2[j,i] < (alpha*dt))
                    a2[i] = 1
                    x2[i] = A
                end
            end
        end

        # total motor number at time step j
		U1 = 0
		U2 = 0

		for i = 1:length(a1)
			U1 = U1 + a1[i]
		end

		for i = 1:length(a2)
			U2 = U2 + a2[i]
		end
        #U1 = sum(a1)
        #U2 = sum(a2)

        # net force 1
        # x*p1*gamma

		F1 = 0
		F2 = 0
		for i = 1:length(x1)
			F1 = F1 + x1[i]
		end

		for i = 1:length(x2)
			F2 = F2 + x2[i]
		end

		F1 = F1*p1g
		F2 = F2*p1g

        #F1 = sum(x1)*p1g
        #F2 = sum(x2)*p1g

        #V += dt*(-F1+F2-zeta*V)/eps
        #V[k+1] = V[k] + dt*(F2-F1-zeta*V[k])/eps
		V = (-F1+F2)/zeta

        if (side == 0) & (V >= switch_v)
        #if (side == 0) & (V[k+1] >= switch_v)
            side = 1
            append!(switch_times,t)
        end

        if (side == 1) & (V <= -switch_v)
        #if (side == 1) & (V[k+1] <= -switch_v)
            side = 0
            append!(switch_times,t)
        end

        #print(V)
        j += 1
        k += 1
    end

    return V,switch_times

end

#using ProfileView

#@profview run_agents(0.01)
#@profview run_agents(0.0001)


#for dt in [0.001 0.0001 0.00001 0.000001 0.0000001]
for dt in [0.000003]
#for dt in [0.0000075 0.0000005 0.00000025 0.0000001]
#for dt in [0.00000005 0.00000002]

	mfpt_dir = string(mfpt_name)

	if ~isdir(mfpt_dir)
		#println("WARNING: DIRECTORY",mfpt_dir," DOES NOT EXIST. PLEASE CREATE.")
		println("creating directory ",mfpt_dir)
		mkdir(mfpt_dir)
	end

	V,out = run_agents(dt)
	times = diff(out[out.>5])

	if false
		plt = plot!(V,show=true)
		gui(plt)
		savefig(plt,"temp.png")
	end

	#fname_tint = "mfpt/TEMP_time_intervals_Tf=%d_nX=%d_nY=%d_seed=%d_dt=%g.txt"%(Tf,nX,nY,seed,round_to_n(dt))

	#@sprintf("%.g",mfpt_dir)

	open(@sprintf("%s/time_intervals_Tf=%d_nX=%d_nY=%d_seed=%d_dt=%.1e_alpha=%g_beta=%g.txt",mfpt_dir,Tf,nX,nY,seed,dt,alpha,beta),"w") do io
		writedlm(io,times)
	end

	#println(times)
	println("__MFPT__=",mean(times),", total switches=",length(times),", dt=",dt)


end

#end
#CSV.write(fname,diff(out))
