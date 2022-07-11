function solve_ipopt(opf)
    return ipopt(opf; tol=1e-5, print_level=0)
end
