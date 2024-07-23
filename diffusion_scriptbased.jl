using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots


@init_parallel_stencil(Threads, Float64, 2)


function square(i, j, ci, cj, di, dj)
    if (abs(i - ci) < di) & (abs(j - cj) < dj)
        return 1
    else
        return 0
    end
end


@parallel function diffusion_2D_step!(C_next, C, D, dt, dx, dy)
	@inn(C_next) = @inn(C) + dt * (D * (@d2_xi(C)/dx^2 + @d2_yi(C)/dy^2))
	return
end



function diffusion_2D()
    lx = 100 			# x dimension in mm
    ly = 20  			# y dimension in mm
    D = 2.0 * 10^-3 	# Diffusion coefficient in mm^2/s

    nx = 1000 # gridsteps x dim
    ny = 200  # gridsteps y dim
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    total_timesteps = 10000

    C = @zeros(nx, ny)
    Cp = @zeros(nx, ny)

    # create initial conditions
    center_x = nx ÷ 2
    center_y = ny ÷ 2
    initial_square_dx = 50
    initial_square_dy = 50
    initial = zeros(Float64, size(C))
    for i in 1:nx
        for j in 1:ny
            initial[i, j] = square(i, j, center_x, center_y, initial_square_dx, initial_square_dy)
        end
    end

    C .+= initial
    
    plt = plot()
    heatmap!(initial)
    gui(plt)

    history = Array{Float64}(undef, total_timesteps, size(C)...)

    # Time loop
    dt = min(dx^2,dy^2)/D/8.1
    for it = 1:total_timesteps
        @parallel diffusion_2D_step!(Cp, C, D, dt, dx, dy)
        C = Cp
        Cp = C

        # save a timestep of the 2D concentration distribution to the history Array
        history[it, :, :] = copy(Cp)
    end
    return history
end




function main()
    println("Hello World from Main!")
    hist = diffusion_2D()

    total_timesteps = size(hist, 1)

    println("Animating ...")
    
    for ti in 1:50:total_timesteps
        heatmap!(hist[ti, :, :])
        savefig("./images/diff-img-$ti.png")
    end

    println("Finished ...")
    s = readline()
end

main()