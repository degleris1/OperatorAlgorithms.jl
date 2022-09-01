# using SparseArrays, SuiteSparse

# z = OperatorAlgorithms.initialize(P)
# F = P._qr
# R = F.R

# x = rand(length(z.primal))
# y = rand(length(z.dual))

# @time Qb = BlockyHouseholderQ(F.Q, 64)
# @show norm(Qb'*x - (F.Q'*x)[1:length(z.dual)])
# @show norm(Qb*y - F.Q*[y; zeros(length(z.primal) - length(z.dual))])

# println("--")

# @time Qb'*x
# @time Qb*y

# @time F.Q*x
# @time F.Q'*x

# println()