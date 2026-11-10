using NonconvexAugLagLab, Test

@testitem "solver capabilities" begin
    using NonconvexAugLagLab, Test
    caps = solver_capabilities(AugLagAlg())
    @test caps.linear_eq === true
    @test caps.nonlinear_eq === true
    @test caps.linear_ineq === true
    @test caps.nonlinear_ineq === true
    @test caps.linear_sd === true
    @test caps.nonlinear_sd === true
    @test caps.box === true
    @test caps.integer === false
    @test caps.infinite_bounds === true
end
